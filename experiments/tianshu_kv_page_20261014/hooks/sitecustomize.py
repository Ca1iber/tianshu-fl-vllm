# SPDX-License-Identifier: Apache-2.0
"""Preserve validated native-prefill composition while allowing KV pages16/32."""
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
ENABLED = os.environ.get('TIANSHU_KV_PAGE_PROBE') == '1'
PROFILE = os.environ.get('TIANSHU_KV_PAGE_BASE_PROFILE', 'S4')
assert PROFILE in ('S4','S8')
base = BASE / ('tianshu_kv16_segments_20261014' if PROFILE == 'S8' else 'tianshu_decode_kv16_w2_20261014') / 'hooks/sitecustomize.py'
expected = '71b8368cca267314481f11e4efe5556b083ca5e86d6e80380ba4ab8632b54a19' if PROFILE == 'S8' else 'bc29613de04666631bf9836007a5d2cc044b294af7c7f5320582de8705dc36c2'
assert hashlib.sha256(base.read_bytes()).hexdigest() == expected
runpy.run_path(str(base))


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        # This loader is the existing NativePrefillLoader, including its L0 router.
        self.original.exec_module(module)
        helper = BASE / 'tianshu_kv_page_20261014/geometry_guard.py'
        spec = importlib.util.spec_from_file_location('_tianshu_page_geometry', helper)
        geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geometry)
        raw = Path(module.__file__).read_bytes()
        wide, provenance, original, candidate = geometry.install(module, raw)
        root = Path(os.environ['TIANSHU_KV_PAGE_RECORD_DIR'])
        root.mkdir(parents=True,exist_ok=True)
        (root / f'original_supported_{os.getpid()}.py').write_text(original)
        (root / f'expanded_supported_{os.getpid()}.py').write_text(candidate)
        provenance.update(base_profile=PROFILE, base_hook_sha256=expected,
            hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
            native_prefill_loader_preserved=True, installed_sources_modified=False)
        (root / f'configured_{os.getpid()}.json').write_text(json.dumps(provenance,indent=2)+'\n')

        def first_supported(kwargs):
            accepted = wide(kwargs)
            if accepted:
                witness = dict(page_size=kwargs['k'].shape[1], k_shape=list(kwargs['k'].shape),
                    q_shape=list(kwargs['q'].shape), base_profile=PROFILE,
                    expanded_predicate_sha256=provenance['expanded_predicate_sha256'])
                (root / f'actual_geometry_{os.getpid()}.json').write_text(json.dumps(witness,indent=2)+'\n')
                # Subsequent layers have exactly the expanded predicate, no recorder wrapper.
                module.supported = wide
            return accepted
        module.supported = first_supported


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm_fl.worker.iluvatar_hybrid_attention':
            return None
        parents = [f for f in sys.meta_path if type(f).__name__ == 'NativePrefillFinder']
        assert len(parents) == 1, 'Need the validated L0 native-prefill finder'
        spec = parents[0].find_spec(fullname,path,target)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if ENABLED:
    assert os.environ.get('TIANSHU_KV16_SERVICE_MODE') == 'candidate'
    assert os.environ.get('TIANSHU_NATIVE_PREFILL_PROBE') == '1'
    if PROFILE == 'S8':
        assert os.environ.get('TIANSHU_SEGMENT_POLICY_PROBE') == '1'
    sys.meta_path.insert(0,Finder())
