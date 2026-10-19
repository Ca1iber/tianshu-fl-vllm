# SPDX-License-Identifier: Apache-2.0
"""Connect guarded byte storage to the actual FL runner and its memory profile."""
import atexit
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent


def install(worker):
    import torch
    from vllm.platforms import current_platform
    from vllm_fl.worker.model_runner import ModelRunnerFL
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    import vllm.v1.attention.backends.triton_attn as backend
    from vllm_fl.worker import iluvatar_adaptive_decode as adaptive
    from vllm_fl.worker import iluvatar_hybrid_attention as hybrid
    from layout_spec import convert_specs

    mode = os.environ['TIANSHU_SOFT8_SERVICE_MODE']
    assert mode in ('control', 'candidate')
    root = Path(os.environ['TIANSHU_SOFT8_RECORD_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    record = dict(mode=mode, enabled=mode == 'candidate', specs=[], cache_views=[],
                  profile_calls=0, availability=[], Python_dispatch_calls=0,
                  Python_counts_not_graph_replay_counts=True)

    def save():
        if mode == 'candidate':
            record.update(producer_records=state['producer'].records,
                dense_records=state['dense'].records, dense_counts=state['dense_counts'],
                byte_counts=state['counts'], scratch_reservation=state['dense'].reservation)
        target = root / f'engine_{os.getpid()}.json'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(target)

    if mode == 'control':
        save()
        return
    assert getattr(current_platform, 'vendor_name', None) == 'iluvatar'
    cfg = worker.vllm_config
    hf = cfg.model_config.hf_config
    assert cfg.model_config.dtype == torch.bfloat16 and cfg.model_config.quantization is None
    assert cfg.speculative_config is None and cfg.lora_config is None
    assert (hf.hidden_size, hf.num_hidden_layers, hf.num_attention_heads, hf.num_key_value_heads) == (2048, 42, 16, 2)
    assert cfg.parallel_config.tensor_parallel_size == cfg.parallel_config.pipeline_parallel_size == cfg.parallel_config.data_parallel_size == 1
    assert cfg.cache_config.cache_dtype == 'auto' and cfg.cache_config.block_size in (None, 16)
    assert cfg.cache_config.kv_cache_memory_bytes is None
    bindings = json.loads((BASE / 'ENGINE_BINDINGS_32g.json').read_text())
    for name, expected in bindings.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    assert not getattr(ModelRunnerFL, '_tianshu_soft8_engine', False)
    import source
    import routing
    reader, writer, proof = source.load_modules(
        Path(sys.modules['tianshu_native_cache_policy'].__file__),
        Path(inspect.getsourcefile(backend.triton_reshape_and_cache_flash)),
        root / f'codec_generated_{os.getpid()}')
    state = routing.install(backend, hybrid, adaptive, reader, writer)
    record.update(codec_proof=proof, bindings=bindings,
                  cache_dtype_config='auto', model_dtype='torch.bfloat16',
                  prefix_cache=cfg.cache_config.enable_prefix_caching,
                  max_model_len=cfg.model_config.max_model_len)

    original_specs = ModelRunnerFL.get_kv_cache_spec
    original_reshape = ModelRunnerFL._reshape_kv_cache_tensors
    original_profile = ModelRunnerFL.profile_run
    original_available = type(worker).determine_available_memory

    @functools.wraps(original_specs)
    def specs(self):
        original = original_specs(self)
        converted = convert_specs(original, FullAttentionSpec, torch.bfloat16, torch.uint8)
        record['specs'].append(dict(layers=len(converted),
            dtypes=sorted({str(s.dtype) for s in converted.values()}),
            before_page_bytes=sorted({s.page_size_bytes for s in original.values()}),
            after_page_bytes=sorted({s.page_size_bytes for s in converted.values()})))
        save()
        return converted

    @functools.wraps(original_reshape)
    def reshape(self, *args, **kwargs):
        views = original_reshape(self, *args, **kwargs)
        assert len(views) == 42
        rows = []
        for name, view in views.items():
            assert view.dtype == torch.uint8 and tuple(view.shape[1:]) == (2, 16, 2, 128)
            assert tuple(view.stride()[1:]) == (4096, 256, 128, 1)
            assert view.stride(0) >= 8192
            rows.append(dict(layer=name, dtype=str(view.dtype), shape=list(view.shape),
                strides=list(view.stride()), logical_bytes=view.numel() * view.element_size()))
        record['cache_views'].append(rows)
        save()
        return views

    @functools.wraps(original_profile)
    def profile(self, *args, **kwargs):
        record['profile_calls'] += 1
        if state['dense'].storage is None:
            # First profile_run is called inside WorkerFL's memory_profiling_fl.
            stack = [f.function for f in inspect.stack()[:8]]
            assert 'determine_available_memory' in stack, stack
            state['dense'].reserve(self.device)
            record['reservation_stack'] = stack
        save()
        return original_profile(self, *args, **kwargs)

    @functools.wraps(original_available)
    def available(self, *args, **kwargs):
        value = original_available(self, *args, **kwargs)
        assert state['dense'].reservation is not None and value > 0
        record['availability'].append(dict(KV_memory_bytes=value,
            torch_memory_allocated=torch.cuda.memory_allocated(),
            arena_reserved_inside_profile=True))
        save()
        return value

    original_dispatch = hybrid.dispatch

    def observed_dispatch(original, route, kw):
        old_sizes = len(state['producer'].records), len(state['dense'].records)
        answer = original_dispatch(original, route, kw)
        record['Python_dispatch_calls'] += 1
        if old_sizes != (len(state['producer'].records), len(state['dense'].records)) or record['Python_dispatch_calls'] % 4096 == 0:
            save()
        return answer

    ModelRunnerFL.get_kv_cache_spec = specs
    ModelRunnerFL._reshape_kv_cache_tensors = reshape
    ModelRunnerFL.profile_run = profile
    ModelRunnerFL._tianshu_soft8_engine = True
    type(worker).determine_available_memory = available
    hybrid.dispatch = observed_dispatch
    atexit.register(save)
    save()
