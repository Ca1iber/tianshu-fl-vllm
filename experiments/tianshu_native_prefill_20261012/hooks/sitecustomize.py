# SPDX-License-Identifier: Apache-2.0
"""Process-only native GQA Prefill with explicit zero no-sink normalization."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / 'tianshu_fa2_tiles_20261012/hooks/sitecustomize.py'))
HYBRID_SHA = '5c55a8d57e438b36d176555fd9273ab9ba665dada4c57d57c19633394c2ed4e4'


class NativePrefillLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == HYBRID_SHA
        import vllm.v1.attention.ops.triton_unified_attention as native
        source = BASE / 'tianshu_native_prefill_20261012/zero_initialization.py'
        spec = importlib.util.spec_from_file_location('tianshu_prefill_zero_generator', source)
        generator = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = generator
        spec.loader.exec_module(generator)
        record_dir = Path(os.environ['TIANSHU_NATIVE_PREFILL_RECORD_DIR'])
        candidate, provenance = generator.load_kernel(native, record_dir / 'source_snapshot' / str(os.getpid()))
        original_dispatch = module.dispatch
        original_kernel = native.kernel_unified_attention
        recorded = set()
        configuration = dict(BLOCK_M=256, TILE_SIZE=128, num_warps=16, num_stages=1)

        class Router:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    assert not args and kwargs['IS_3D'] is False
                    assert kwargs['num_queries_per_kv'] == 8 and kwargs['HEAD_SIZE'] == 128
                    block_q = configuration['BLOCK_M'] // kwargs['num_queries_per_kv']
                    new_grid = (kwargs['query_ptr'].shape[0] // block_q + kwargs['num_seqs'], grid[1])
                    kwargs.update(configuration, BLOCK_Q=block_q)
                    return candidate[new_grid](**kwargs)
                return launch
        router = Router()

        def dispatch(original, route, kwargs):
            if route is None or kwargs['max_seqlen_q'] <= 1 or not module.supported(kwargs):
                return original_dispatch(original, route, kwargs)
            nd = route.num_decodes
            if nd:
                decode = dict(kwargs)
                decode.update(q=kwargs['q'][:nd], out=kwargs['out'][:nd],
                    cu_seqlens_q=kwargs['cu_seqlens_q'][:nd + 1], max_seqlen_q=1,
                    seqused_k=kwargs['seqused_k'][:nd], block_table=kwargs['block_table'][:nd])
                original(**decode)
                module.COUNTS['mixed'] += 1
            else:
                module.COUNTS['prefill'] += 1
            prefill = dict(kwargs)
            prefill.update(q=kwargs['q'][nd:], out=kwargs['out'][nd:],
                cu_seqlens_q=route.prefill_cu, max_seqlen_q=route.max_prefill_query,
                seqused_k=kwargs['seqused_k'][nd:], block_table=kwargs['block_table'][nd:])
            previous = native.kernel_unified_attention
            native.kernel_unified_attention = router
            try:
                native.unified_attention(**prefill)
            finally:
                native.kernel_unified_attention = previous
            key = prefill['q'].shape[0], prefill['seqused_k'].numel(), route.max_prefill_query
            if key not in recorded:
                recorded.add(key)
                record_dir.mkdir(parents=True, exist_ok=True)
                (record_dir / f'native_prefill_{os.getpid()}_{key[0]}_{key[1]}_{key[2]}.json').write_text(json.dumps({
                    'pid': os.getpid(), 'configuration': configuration,
                    'prefill_query_tokens': key[0], 'prefill_sequences': key[1], 'max_query_len': key[2],
                    'decode_prefix_sequences': nd, 'hybrid_sha256': HYBRID_SHA,
                    'kernel_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'generated_kernel_sha256': provenance['generated_sha256'],
                    'zero_initialization': provenance, 'decode_arithmetic_changed': False}, indent=2) + '\n')
            return None
        module.dispatch = dispatch


class NativePrefillFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm_fl.worker.iluvatar_hybrid_attention':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = NativePrefillLoader(spec.loader)
        return spec


if os.environ.get('TIANSHU_NATIVE_PREFILL_PROBE') == '1':
    sys.meta_path.insert(0, NativePrefillFinder())
