# SPDX-License-Identifier: Apache-2.0
"""Candidate hook installed after the frozen GEO profile hooks."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys

FOLDER = Path(__file__).resolve().parents[1]
import runpy
runpy.run_path(str(FOLDER.parent / "tianshu_gemm_topp_geo_20261016/hooks/sitecustomize.py"))


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        import torch
        from vllm.platforms import current_platform
        assert current_platform.vendor_name == 'iluvatar'
        geo = sys.modules['_tianshu_topp_geometric_geo']
        spec = importlib.util.spec_from_file_location('_tianshu_tiny_topp_api', FOLDER / 'api_route.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        root = Path(os.environ['TIANSHU_TINY_TOPP_RECORD_DIR'])
        root.mkdir(parents=True, exist_ok=True)
        observed = dict(python_mask_calls=0, batches={}, compiled={},
                        scope='Python launches only; CUDA Graph replay does not rerun observer')

        def save():
            path = root / f'actual_{os.getpid()}.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(observed, indent=2) + '\n')
            temporary.replace(path)

        def observe(logits):
            observed['python_mask_calls'] += 1
            batch = str(logits.shape[0])
            fresh = batch not in observed['batches']
            observed['batches'][batch] = observed['batches'].get(batch, 0) + 1
            if fresh or observed['python_mask_calls'] % 256 == 0:
                save()

        def compiled(kwargs, kernel):
            batch = str(kwargs['BATCH_SIZE'])
            if batch not in observed['compiled']:
                observed['compiled'][batch] = dict(
                    BLOCK_SIZE=kwargs['BLOCK_SIZE'], BLOCK_SIZE_TRUNC=kwargs['BLOCK_SIZE_TRUNC'],
                    num_warps=kernel.metadata.num_warps, num_stages=kernel.metadata.num_stages,
                    n_regs=getattr(kernel, 'n_regs', None), n_spills=getattr(kernel, 'n_spills', None),
                    actual_source_sha256=helper.GEO_SHA)
                save()

        record = helper.install(module, geo, torch, observer=observe, compiled_observer=compiled)
        (root / f'installed_{os.getpid()}.json').write_text(json.dumps(dict(
            pid=os.getpid(), sampler_source_sha256=helper.SAMPLER_SHA,
            generated_source_sha256=helper.GEO_SHA,
            api_route_sha256=hashlib.sha256((FOLDER / 'api_route.py').read_bytes()).hexdigest(),
            hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            before_export_to_import_aliases=True, scope=record['scope'],
            callback_installed=True, actual_GPU_calls_observed=False,
            fresh_Math=False, adopted=False), indent=2) + '\n')


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm.v1.sample.ops.topk_topp_sampler':
            return None
        # Preserve any already configured loader, and avoid self-delegation.
        index = sys.meta_path.index(self)
        sys.meta_path.pop(index)
        try:
            spec = None
            for finder in tuple(sys.meta_path):
                method = getattr(finder, 'find_spec', None)
                if method is not None:
                    spec = method(fullname, path, target)
                    if spec is not None:
                        break
        finally:
            sys.meta_path.insert(index, self)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.environ.get('TIANSHU_TINY_TOPP_CANDIDATE') == '1':
    assert 'vllm.v1.sample.ops.topk_topp_sampler' not in sys.modules
    assert 'vllm.v1.sample.rejection_sampler' not in sys.modules
    sys.meta_path.insert(0, Finder())
