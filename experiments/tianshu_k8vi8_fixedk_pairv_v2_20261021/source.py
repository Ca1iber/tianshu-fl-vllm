# SPDX-License-Identifier: Apache-2.0
"""Factorial V8 probe: retain native path, constant K-scale, paired V8 load."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_k8vi8_heads_v2_20261021/source.py'
spec=importlib.util.spec_from_file_location('_fixedk_V8_native_source',PARENT)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
SOURCE=parent.SOURCE
SOURCE_SHA=parent.SOURCE_SHA

def render(raw):
    original,parent_proof=parent.render(raw);text=original
    signature='def kernel_unified_attention(\n'
    new_signature=signature+'    FIXED_SOFT8_K: tl.constexpr,\n    PAIR_V_LOAD: tl.constexpr,\n'
    old_scale='''            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )'''
    new_scale='''            if FIXED_SOFT8_K:
                tl.static_assert(KV_QUANT_MODE == 2 and key_cache_ptr.dtype.element_ty == tl.uint8)
                k_token_head_scales = tl.full((TILE_SIZE,), 1.0, tl.float32)
            else:
                k_token_head_scales = tl.load(
                    k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
                )'''
    old_v='''            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
                cache_modifier=KV_CACHE_MODIFIER if IS_3D else "",
            )'''
    new_v='''            if PAIR_V_LOAD:
                tl.static_assert(KV_QUANT_MODE == 2 and HEAD_SIZE == 128 and HEAD_SIZE_PADDED == 128)
                tl.static_assert(value_cache_ptr.dtype.element_ty == tl.int8 and stride_v_cache_3 == 1)
                v_pair_offset = (
                    physical_block_idx[:, None] * (stride_v_cache_0 // 2)
                    + kv_head_idx * (stride_v_cache_2 // 2)
                    + tl.arange(0, 64)[None, :]
                    + (seq_offset % BLOCK_SIZE)[:, None] * (stride_v_cache_1 // 2)
                )
                v_pair = tl.load(value_cache_ptr.to(tl.pointer_type(tl.uint16)) + v_pair_offset,
                    mask=tile_mask[:, None], other=0, cache_modifier=KV_CACHE_MODIFIER if IS_3D else "")
                v_low = (v_pair & 255).to(tl.int8)
                v_high = (v_pair >> 8).to(tl.int8)
                V_load = tl.join(v_low, v_high).reshape(TILE_SIZE, 128)
            else:
                V_load = tl.load(
                    value_cache_ptr + v_offset,
                    mask=dim_mask[None, :] & tile_mask[:, None],
                    other=0.0,
                    cache_modifier=KV_CACHE_MODIFIER if IS_3D else "",
                )'''
    edits=[(signature,new_signature),(old_scale,new_scale),(old_v,new_v)]
    for before,after in edits:assert text.count(before)==1;text=text.replace(before,after)
    restored=text
    for before,after in reversed(edits):assert restored.count(after)==1;restored=restored.replace(after,before)
    assert restored==original
    def function(src,name):return next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('_cast_kv_tile','reduce_segments','unified_attention'):
        assert ast.dump(function(original,name))==ast.dump(function(text,name))
    return text,dict(parent=parent_proof,three_exact_reversible_changes=True,
        generated_sha256=hashlib.sha256(text.encode()).hexdigest(),original_V8_reader_AST_API_reducer_and_cast_unchanged=True,
        fixed_K_scale_requires_actual_writer_all_one_proof=True,paired_V8_requires_even_aligned_strides_and_D128=True,
        softmax_and_P_scaling_arithmetic_unchanged=True)

def load_modules(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    text,proof=render(SOURCE.read_bytes());file=root/'generated_reader.py';file.write_text(text)
    name='_tianshu_k8vi8_fixedk_pairv_reader';spec=importlib.util.spec_from_file_location(name,file)
    reader=importlib.util.module_from_spec(spec);sys.modules[name]=reader;spec.loader.exec_module(reader)
    return reader,dict(proof,generated_source=str(file))
