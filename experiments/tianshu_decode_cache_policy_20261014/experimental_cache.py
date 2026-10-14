# SPDX-License-Identifier: Apache-2.0
"""Generate a checked L0 copy with explicit 3D K/V load cache policies."""
import hashlib
import importlib.util
from pathlib import Path
import sys


L0_SHA='4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e'


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest()==L0_SHA
    text=raw.decode()
    signature='    Q_IS_FP8: tl.constexpr = False,\n):'
    assert text.count(signature)==1
    text=text.replace(signature,'    Q_IS_FP8: tl.constexpr = False,\n    KV_CACHE_MODIFIER: tl.constexpr = "",\n):')
    for pointer,mask in (('key_cache_ptr + k_offset','dim_mask[:, None] & tile_mask[None, :]'),
                         ('value_cache_ptr + v_offset','dim_mask[None, :] & tile_mask[:, None]')):
        before=f'''                {pointer},
                mask={mask},
                other=0.0,
'''
        assert text.count(before)==1
        text=text.replace(before,before+'                cache_modifier=KV_CACHE_MODIFIER if IS_3D else "",\n')
    return text


def load_kernel(l0_source,directory):
    raw=Path(l0_source).read_bytes();text=render_source(raw)
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    output=root/'generated_cache_policy.py';output.write_text(text)
    name='tianshu_native_cache_policy'
    spec=importlib.util.spec_from_file_location(name,output)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.kernel_unified_attention,dict(original_L0_sha256=L0_SHA,generated_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        generated_source=str(output),attention_arithmetic_changed=False,only_3D_KV_load_cache_policy_changed=True,
        installed_sources_modified=False)
