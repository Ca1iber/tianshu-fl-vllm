# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual installed Breakable wrapper/capture AST with CPU-only stand-ins and strict forwarding checks."""
import argparse
import ast
from contextlib import contextmanager, ExitStack
from enum import Enum
import hashlib
import weakref
import threading
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
classes = [next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    for name in ('BreakableCUDAGraphCapture','BreakableCUDAGraphWrapper')]


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
    def capture_begin(self,pool=None):
        assert not capture[0]
        capture[0]=True
    def capture_end(self):
        assert capture[0]
        capture[0]=False
    def replay(self):
        calls['replay']+=1


torch=SimpleNamespace(Tensor=type('UnusedTensor',(),{}),bfloat16='BF16',uint8='U8',
    accelerator=SimpleNamespace(empty_cache=lambda:None),
    cuda=SimpleNamespace(CUDAGraph=Graph,is_current_stream_capturing=lambda:capture[0]))
offloader=SimpleNamespace(sync_prev_onload=lambda:None,join_after_forward=lambda:None)
namespace=dict(is_forward_context_available=lambda:available[0],get_forward_context=lambda:context,
    CUDAGraphMode=Mode,_BreakableEntry=lambda batch_descriptor:SimpleNamespace(batch_descriptor=batch_descriptor,capture=None,output=None,input_addresses=None),
    validate_cudagraph_capturing_enabled=lambda:None,torch=torch,weakref=weakref,threading=threading,
    gc=SimpleNamespace(collect=lambda:None),envs=SimpleNamespace(VLLM_LOGGING_LEVEL='INFO'),
    logger=SimpleNamespace(info_once=lambda *a:None,debug=lambda *a:None),
    current_platform=SimpleNamespace(get_global_graph_pool=lambda:None,graph_pool_handle=lambda:'pool'),
    set_graph_pool_id=lambda value:None,get_offloader=lambda:offloader,weak_ref_tensors=lambda value:value)
future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
module_ast=ast.fix_missing_locations(ast.Module(body=[future,*classes],type_ignores=[]))
exec(compile(module_ast,str(a.graph_source),'exec'),namespace)
Wrapper=namespace['BreakableCUDAGraphWrapper']
Capture=namespace['BreakableCUDAGraphCapture']
graph_module=SimpleNamespace(__file__=str(a.graph_source),BreakableCUDAGraphWrapper=Wrapper,
    is_forward_context_available=lambda:available[0],get_forward_context=lambda:context,CUDAGraphMode=Mode,torch=torch)
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


wrapper = Wrapper(runnable,SimpleNamespace(compilation_config=None))


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
assert calls == dict(runnable=3, kernel=3, replay=2)
assert len(trace.STATE['graphs']) == 1
edge = next(iter(trace.STATE['graphs'].values()))
assert edge['capture_calls'] == 1 and edge['replay_calls'] == 2
assert edge['segments']['GPU_graphs']==1 and edge['segments']['eager_breaks']==0
assert edge['frames']['64'] == dict(count=2, min_computed=8192, max_computed=8192)
captured = [v for v in trace.STATE['launches'].values() if v['actually_inside_capture']]
assert len(captured) == 1 and captured[0]['calls'] == 1 and captured[0]['max_max_seqlen_k'] == 16384
assert captured[0]['graph_key'] in trace.STATE['graphs']
assert all(trace.STATE[name] is None for name in ('active_frame', 'active_graph', 'active_call', 'active_action'))
# Execute the actual add_eager/_begin/_end/replay flow as well.
context.batch_descriptor='B64-with-eager-break'
context.cudagraph_runtime_mode=Mode.FULL
def eager_runnable(*args,**kwargs):
    assert args==(sentinel,) and kwargs==dict(unchanged=keyword)
    return Capture.current().add_eager(lambda:hybrid.dispatch(sentinel,None,dict(max_seqlen_k=16384)))
wrapper.runnable=eager_runnable
assert runner._model_forward(sentinel,unchanged=keyword) is output
edge2=[e for e in trace.STATE['graphs'].values() if 'with-eager-break' in e['descriptor']][0]
assert edge2['segments']['GPU_graphs']==2 and edge2['segments']['eager_breaks']==1
assert runner._model_forward(sentinel,unchanged=keyword) is output
records=[x for x in trace.STATE['launches'].values() if x['wrapper_action']=='replay']
assert records and all(not x['actually_inside_capture'] for x in records)
assert all(trace.STATE[name] is None for name in ('active_frame','active_graph','active_call','active_action'))
context.batch_descriptor='expected-failure'
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
assert all(trace.STATE[name] is None for name in ('active_frame', 'active_graph', 'active_call', 'active_action'))
for file in base.rglob('*.py'):
    ast.parse(file.read_text())
result = dict(complete=True, CPU_only=True, actual_installed_Graph_call_AST_executed=True,
    capture_replay_NONE_no_context_PIECEWISE_accepted_and_exception_paths_checked=True, actual_add_eager_and_segment_replay_AST_checked=True,
    original_argument_and_return_identity_preserved=True, original_exception_identity_preserved=True,
    graph_edge_to_captured_configuration_link_checked=True, computed_lengths_recorded_without_tensor_reads=True,
    input_sources={str(a.graph_source):trace.EXPECTED['graph'], str(a.runner_source):trace.EXPECTED['runner']},
    sources={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(base.rglob('*.py'))},
    GPU_not_run=True, performance_and_quality_not_measured=True, adopted=False, actual_Breakable_wrapper_and_capture_AST_executed=True)
(base/'CPU_GATE_32g.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
