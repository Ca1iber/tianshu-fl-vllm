# SPDX-License-Identifier: Apache-2.0
"""Separate K/V cache and eviction hints; retain exact arithmetic and reverse to current CV source."""
import hashlib
import importlib.util
from pathlib import Path
import sys

CV_SHA = "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
POLICIES = [
    (".cv", ".cv", "", ""), (".cv", "", "", ""), ("", ".cv", "", ""),
    (".cv", ".cg", "", ""), (".cg", ".cv", "", ""),
    (".cv", ".cv", "evict_first", "evict_first"), (".cv", ".cv", "evict_last", "evict_last"),
    (".cv", ".cv", "evict_last", "evict_first"), (".cv", ".cv", "evict_first", "evict_last"),
]


def configuration(policy):
    return dict(zip(("K_CACHE_MODIFIER", "V_CACHE_MODIFIER", "K_EVICTION", "V_EVICTION"), policy))


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest() == CV_SHA
    text = raw.decode()
    edits = []
    before = '    KV_CACHE_MODIFIER: tl.constexpr = "",\n):'
    after = '    KV_CACHE_MODIFIER: tl.constexpr = "",\n    K_CACHE_MODIFIER: tl.constexpr = ".cv",\n    V_CACHE_MODIFIER: tl.constexpr = ".cv",\n    K_EVICTION: tl.constexpr = "",\n    V_EVICTION: tl.constexpr = "",\n):'
    assert text.count(before) == 1
    text = text.replace(before, after)
    edits.append((before, after))
    for pointer, mask, prefix in [("key_cache_ptr + k_offset", "dim_mask[:, None] & tile_mask[None, :]", "K"),
                                  ("value_cache_ptr + v_offset", "dim_mask[None, :] & tile_mask[:, None]", "V")]:
        before = f'                {pointer},\n                mask={mask},\n                other=0.0,\n                cache_modifier=KV_CACHE_MODIFIER if IS_3D else "",\n'
        after = before.replace('cache_modifier=KV_CACHE_MODIFIER if IS_3D else "",',
                               f'cache_modifier={prefix}_CACHE_MODIFIER if IS_3D else "",\n                eviction_policy={prefix}_EVICTION if IS_3D else "",')
        assert text.count(before) == 1
        text = text.replace(before, after)
        edits.append((before, after))
    restored = text
    for before, after in reversed(edits):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored.encode() == raw
    return text


def load_kernel(source, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "generated_asymmetric_cache.py"
    path.write_text(render_source(Path(source).read_bytes()))
    name = "_tianshu_asymmetric_cache"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.kernel_unified_attention, dict(original_CV_sha256=CV_SHA,
        generated_source=str(path), generated_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        exactly_reversible_to_current_CV_source=True, attention_arithmetic_unchanged=True,
        installed_sources_modified=False, only_3D_key_value_load_hints_changed=True)
