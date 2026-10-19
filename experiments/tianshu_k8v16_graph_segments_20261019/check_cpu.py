# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual installed Graph callback AST with CPU-only stand-ins and strict forwarding checks."""
import argparse
import ast
from contextlib import contextmanager, ExitStack
from enum import Enum
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

base = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--graph-source', type=Path, required=True)
p.add_argument('--runner-source', type=Path, required=True)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('_graph_segments_CPU_trace', base/'trace.py')
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)
assert hashlib.sha256(a.graph_source.read_bytes()).hexdigest() == trace.EXPECTED['graph']
assert hashlib.sha256(a.runner_source.read_bytes()).hexdigest() == trace.EXPECTED['runner']
tree = ast.parse(a.graph_source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CUDAGraphWrapper')
node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')


class Mode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2


capture = [False]
available = [True]
context = SimpleNamespace(batch_descriptor='B64', cudagraph_runtime_mode=Mode.FULL)
calls = dict(runnable=0, kernel=0, replay=0)
sentinel, keyword, output = object(), object(), object()


class Graph:
    def replay(self):
        calls['replay'] += 1


@contextmanager
def graph_context(*args, **kwargs):
    assert not capture[0]
    capture[0] = True
    try:
        yield
    finally:
        capture[0] = False


torch = SimpleNamespace(Tensor=type('UnusedTensor', (), {}), bfloat16='BF16', uint8='U8',
    cuda=SimpleNamespace(CUDAGraph=Graph, graph=graph_context, is_current_stream_capturing=lambda: capture[0]))
offloader = SimpleNamespace(sync_prev_onload=lambda: None, join_after_forward=lambda: None)
namespace = dict(Any=object, is_forward_context_available=lambda: available[0],
    get_forward_context=lambda: context, CUDAGraphMode=Mode,
    CUDAGraphEntry=lambda batch_descriptor: SimpleNamespace(batch_descriptor=batch_descriptor, cudagraph=None),
    validate_cudagraph_capturing_enabled=lambda: None, torch=torch, ExitStack=ExitStack,
    current_platform=SimpleNamespace(graph_pool_handle=lambda: 'pool'), set_graph_pool_id=lambda value: None,
    get_offloader=lambda: offloader, current_stream=lambda: None, weak_ref_tensors=lambda value: value,
    compilation_counter=SimpleNamespace(num_cudagraph_captured=0))
exec(compile(ast.Module(body=[node], type_ignores=[]), str(a.graph_source), 'exec'), namespace)


class Wrapper:
    __call__ = namespace['__call__']

    def __init__(self, runnable):
        self.runnable, self.runtime_mode = runnable, Mode.FULL
        self.concrete_cudagraph_entries = {}
        self.graph_pool, self.is_debugging_mode = None, False
        self.cudagraph_options = SimpleNamespace(debug_log_enable=False, gc_disable=False, weak_ref_output=False)


graph_module = SimpleNamespace(__file__=str(a.graph_source), CUDAGraphWrapper=Wrapper,
    is_forward_context_available=lambda: available[0], get_forward_context=lambda: context, CUDAGraphMode=Mode)
trace.install_graph(graph_module)


class Producer:
    def __getitem__(self, grid):
        def launch(*args, **kw):
            assert grid == (96, 2, 4) and not args and kw == dict(unchanged=keyword)
            calls['kernel'] += 1
            self.last_record = dict(IS_3D=True, sequences=64, segments=4, tile=16, warps=1, stages=1)
            return output
        return launch


producer = Producer()
hybrid = SimpleNamespace(dispatch=None)


def original_install(backend, target, adaptive, reader, writer):
    def dispatch(original, route, kw):
        assert original is sentinel and route is None and kw == dict(max_seqlen_k=16384)
        return producer[(96, 2, 4)](unchanged=keyword)
    target.dispatch = dispatch
    return output


routing_source = base.parent/'tianshu_k8v16_pipeline_20261019/legacy_routing.py'
routing = SimpleNamespace(__file__=str(routing_source), torch=torch, Producer=Producer, install=original_install)
trace.install_routing(routing)
assert routing.install(None, hybrid, None, None, None) is output


def runnable(*args, **kwargs):
    assert args == (sentinel,) and kwargs == dict(unchanged=keyword)
    calls['runnable'] += 1
    return hybrid.dispatch(sentinel, None, dict(max_seqlen_k=16384))


wrapper = Wrapper(runnable)


class Runner:
    input_batch = SimpleNamespace(num_reqs=64, num_computed_tokens_cpu=[4096]*64)

    def _model_forward(self, *args, **kwargs):
        return wrapper(*args, **kwargs)


runner_module = SimpleNamespace(__file__=str(a.runner_source), ModelRunnerFL=Runner)
trace.install_runner(runner_module)
runner = Runner()
assert runner._model_forward(sentinel, unchanged=keyword) is output  # actual AST capture
runner.input_batch.num_computed_tokens_cpu = [8192]*64
assert runner._model_forward(sentinel, unchanged=keyword) is output  # actual AST replay
assert calls == dict(runnable=1, kernel=1, replay=1)
context.cudagraph_runtime_mode = Mode.NONE
assert runner._model_forward(sentinel, unchanged=keyword) is output
available[0] = False
assert runner._model_forward(sentinel, unchanged=keyword) is output
available[0] = True
context.cudagraph_runtime_mode = Mode.PIECEWISE
assert runner._model_forward(sentinel, unchanged=keyword) is output
assert calls == dict(runnable=4, kernel=4, replay=1)
assert len(trace.STATE['graphs']) == 1
edge = next(iter(trace.STATE['graphs'].values()))
assert edge['capture_calls'] == edge['replay_calls'] == 1
assert edge['frames']['64'] == dict(count=1, min_computed=8192, max_computed=8192)
captured = [v for v in trace.STATE['launches'].values() if v['actually_inside_capture']]
assert len(captured) == 1 and captured[0]['calls'] == 1 and captured[0]['max_max_seqlen_k'] == 16384
assert captured[0]['graph_key'] in trace.STATE['graphs']
assert all(trace.STATE[name] is None for name in ('active_frame', 'active_graph', 'active_call'))
context.cudagraph_runtime_mode = Mode.NONE
error = RuntimeError('forwarding sentinel')
def failed(*args, **kwargs):
    raise error
wrapper.runnable = failed
try:
    runner._model_forward(sentinel, unchanged=keyword)
except RuntimeError as observed:
    assert observed is error
else:
    raise AssertionError('exception swallowed')
assert all(trace.STATE[name] is None for name in ('active_frame', 'active_graph', 'active_call'))
for file in base.rglob('*.py'):
    ast.parse(file.read_text())
result = dict(complete=True, CPU_only=True, actual_installed_Graph_call_AST_executed=True,
    capture_replay_NONE_no_context_mode_mismatch_and_exception_paths_checked=True,
    original_argument_and_return_identity_preserved=True, original_exception_identity_preserved=True,
    graph_edge_to_captured_configuration_link_checked=True, computed_lengths_recorded_without_tensor_reads=True,
    input_sources={str(a.graph_source):trace.EXPECTED['graph'], str(a.runner_source):trace.EXPECTED['runner']},
    sources={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(base.rglob('*.py'))},
    GPU_not_run=True, performance_and_quality_not_measured=True, adopted=False)
(base/'CPU_GATE_32g.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
