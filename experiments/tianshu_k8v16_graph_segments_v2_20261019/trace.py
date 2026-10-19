# SPDX-License-Identifier: Apache-2.0
"""Observe exact Graph edges and their compiled decode policy; forward all calls unchanged."""
import atexit
import functools
import hashlib
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXPECTED = {
    'graph': '722683b44b4461ffd9a6498741da18945cbaac231c3bb01f0b0443dafe9e053e',
    'runner': 'b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e',
    'routing': '9dae2408de9f77e39350cfd70661e4e282ef0d69f798f968c1886d4172a6dc41',
}
STATE = dict(graphs={}, launches={}, sources={}, calls=0, active_graph=None,
    active_call=None, active_frame=None, active_action=None, bound=[], diagnostic_only=True,
    attention_arguments_and_return_values_forwarded=True, RNG_not_read_or_modified=True)


def save():
    value = os.environ.get('TIANSHU_GRAPH_SEGMENTS_RECORD_DIR')
    if not value:
        return
    root = Path(value)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'trace_{os.getpid()}.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps({k:v for k,v in STATE.items() if not k.startswith('active_')}, indent=2) + '\n')
    temp.replace(path)


def source(role, module):
    path = Path(module.__file__)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == EXPECTED[role], (role, path)
    assert role not in STATE['bound']
    STATE['bound'].append(role)
    STATE['sources'][str(path)] = sha


def tick():
    STATE['calls'] += 1
    if STATE['calls'] % 50 == 0:
        save()


def segment_summary(capture, torch):
    result = []
    for index, fn in enumerate(capture.segments):
        owner = getattr(fn, '__self__', None)
        gpu = isinstance(owner, torch.cuda.CUDAGraph)
        entry = dict(index=index, kind='GPU_graph' if gpu else 'eager',
            callable_module=getattr(fn, '__module__', type(fn).__module__),
            callable_name=getattr(fn, '__qualname__', type(fn).__qualname__),
            graph_object_id=id(owner) if gpu else None)
        if not gpu:
            names = []
            for cell in getattr(fn, '__closure__', ()) or ():
                try:value = cell.cell_contents
                except ValueError:continue
                if callable(value):
                    names.append(dict(module=getattr(value, '__module__', type(value).__module__),
                        name=getattr(value, '__qualname__', type(value).__qualname__)))
            entry['closure_callables'] = names
        result.append(entry)
    assert sum(x['kind']=='GPU_graph' for x in result) == capture.num_graphs
    assert sum(x['kind']=='eager' for x in result) == capture.num_eager_breaks
    return dict(GPU_graphs=capture.num_graphs, eager_breaks=capture.num_eager_breaks, entries=result)


def install_graph(module):
    source('graph', module)
    original = module.BreakableCUDAGraphWrapper.__call__

    @functools.wraps(original)
    def call(owner, *args, **kwargs):
        if not module.is_forward_context_available():
            return original(owner, *args, **kwargs)
        context = module.get_forward_context()
        mode, descriptor = context.cudagraph_runtime_mode, context.batch_descriptor
        if mode == module.CUDAGraphMode.NONE:
            return original(owner, *args, **kwargs)
        assert descriptor is not None
        key = str((id(owner), repr(descriptor)))
        entry = owner.entries.get(descriptor)
        action = 'capture' if entry is None or entry.capture is None else 'replay'
        graph = STATE['graphs'].setdefault(key, dict(descriptor=repr(descriptor), runtime_mode=mode.name,
            capture_calls=0, replay_calls=0, frames={}, capture_object_id=None, segments=None))
        old = STATE['active_graph']
        old_action = STATE['active_action']
        STATE['active_graph'] = key
        STATE['active_action'] = action
        try:
            result = original(owner, *args, **kwargs)
            entry = owner.entries[descriptor]
            assert entry.capture is not None
            identity = id(entry.capture)
            assert graph['capture_object_id'] in (None, identity)
            graph['capture_object_id'] = identity
            if graph['segments'] is None:
                graph['segments'] = segment_summary(entry.capture, module.torch)
            else:
                assert graph['segments']['GPU_graphs'] == entry.capture.num_graphs
                assert graph['segments']['eager_breaks'] == entry.capture.num_eager_breaks
            graph[action + '_calls'] += 1
            frame = STATE['active_frame']
            if action == 'replay' and frame is not None:
                item = graph['frames'].setdefault(str(frame['requests']), dict(count=0,
                    min_computed=frame['min_computed'], max_computed=frame['max_computed']))
                item['count'] += 1
                item['min_computed'] = min(item['min_computed'], frame['min_computed'])
                item['max_computed'] = max(item['max_computed'], frame['max_computed'])
            tick()
            if action == 'capture' or (frame is not None and action == 'replay' and item['count'] == 1):
                save()
            return result
        finally:
            STATE['active_graph'] = old
            STATE['active_action'] = old_action

    module.BreakableCUDAGraphWrapper.__call__ = call


def install_runner(module):
    source('runner', module)
    original = module.ModelRunnerFL._model_forward

    @functools.wraps(original)
    def forward(owner, *args, **kwargs):
        batch = owner.input_batch
        n = int(batch.num_reqs)
        frame = None if not n else dict(requests=n,
            min_computed=int(min(batch.num_computed_tokens_cpu[:n])),
            max_computed=int(max(batch.num_computed_tokens_cpu[:n])))
        old = STATE['active_frame']
        STATE['active_frame'] = frame
        try:
            return original(owner, *args, **kwargs)
        finally:
            STATE['active_frame'] = old

    module.ModelRunnerFL._model_forward = forward


def install_routing(module):
    source('routing', module)
    original_get = module.Producer.__getitem__
    original_install = module.install

    def get(owner, grid):
        original_launch = original_get(owner, grid)

        def launch(*args, **kwargs):
            capturing = bool(module.torch.cuda.is_current_stream_capturing())
            result = original_launch(*args, **kwargs)
            actual = dict(owner.last_record)
            context = STATE['active_call']
            max_length = None if context is None else context['max_seqlen_k']
            graph_key = STATE['active_graph']
            action = STATE['active_action']
            key = str((graph_key, action, capturing, actual['IS_3D'], actual['sequences'],
                actual['segments'], actual['tile'], actual['warps'], max_length is not None and max_length > 128))
            item = STATE['launches'].setdefault(key, dict(graph_key=graph_key, wrapper_action=action,
                actually_inside_capture=capturing, configuration=actual, calls=0,
                min_max_seqlen_k=max_length, max_max_seqlen_k=max_length))
            item['calls'] += 1
            if max_length is not None:
                item['min_max_seqlen_k'] = max_length if item['min_max_seqlen_k'] is None else min(item['min_max_seqlen_k'], max_length)
                item['max_max_seqlen_k'] = max_length if item['max_max_seqlen_k'] is None else max(item['max_max_seqlen_k'], max_length)
            if not capturing and item['calls'] == 1:
                save()
            return result

        return launch

    def install(backend, hybrid, adaptive, reader, writer):
        result = original_install(backend, hybrid, adaptive, reader, writer)
        original_dispatch = hybrid.dispatch

        def dispatch(original, route, kw):
            old = STATE['active_call']
            STATE['active_call'] = dict(max_seqlen_k=int(kw['max_seqlen_k']))
            try:
                return original_dispatch(original, route, kw)
            finally:
                STATE['active_call'] = old

        hybrid.dispatch = dispatch
        return result

    module.Producer.__getitem__ = get
    module.install = install


atexit.register(save)
