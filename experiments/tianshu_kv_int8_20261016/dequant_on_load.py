# SPDX-License-Identifier: Apache-2.0
"""Guarded INT8 load-time dequantization, preserving original fallback source."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

CV_SHA='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest()==CV_SHA
    text=raw.decode();replacements=[]
    def replace(before,after):
        nonlocal text
        assert text.count(before)==1,before
        text=text.replace(before,after);replacements.append((before,after))
    replace('    KV_CACHE_MODIFIER: tl.constexpr = "",\n):',
            '    KV_CACHE_MODIFIER: tl.constexpr = "",\n    DEQUANT_KV_ON_LOAD: tl.constexpr = False,\n):')
    anchor='''            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )
'''
    replace(anchor,anchor+'''
        if DEQUANT_KV_ON_LOAD:
            tl.static_assert(KV_QUANT_MODE == 2)
            K = (K_load.to(tl.float32) * k_token_head_scales[None, :]).to(Q.dtype)
            V = (V_load.to(tl.float32) * v_token_head_scales[:, None]).to(Q.dtype)
''')
    replace('''        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale''',
            '''        if USE_PER_TOKEN_HEAD_SCALES and not DEQUANT_KV_ON_LOAD:
            # Per-token-head quant: fuse softmax_scale''')
    replace('''        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale''',
            '''        if USE_PER_TOKEN_HEAD_SCALES and not DEQUANT_KV_ON_LOAD:
            # Per-token-head quant: apply v_scale''')
    ast.parse(text)
    restored=text
    for before,after in reversed(replacements):
        assert restored.count(after)==1;restored=restored.replace(after,before)
    assert restored==raw.decode()
    return text,dict(original_sha256=CV_SHA,generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        reversible_to_original=True,replacements=len(replacements),INT8_only_static_guard=True,
        BF16_dequant_rounding_changes_numerical_path=True,original_fallback_source_preserved=True)


def load_kernel(source,directory):
    generated,record=render_source(Path(source).read_bytes())
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    output=root/'generated_dequant_on_load.py';output.write_text(generated)
    name='_tianshu_int8_dequant_on_load'
    spec=importlib.util.spec_from_file_location(name,output)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    record['generated_source']=str(output)
    return module.kernel_unified_attention,record
