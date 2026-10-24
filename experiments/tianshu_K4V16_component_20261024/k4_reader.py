# SPDX-License-Identifier: Apache-2.0
"""Transform only the current paired-K attention kernel; wrapper/reducer unchanged."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
PARENT_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'

def render(parent):
    original=Path(parent).read_text()
    assert hashlib.sha256(original.encode()).hexdigest()==PARENT_SHA
    anchor='    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1 and Q_IS_FP8\n'
    guard='''    tl.static_assert(IS_3D and KV_QUANT_MODE == 2)
    tl.static_assert(HEAD_SIZE == 128 and HEAD_SIZE_PADDED == 128)
    tl.static_assert(not USE_TD and not USE_TD_QO)
    tl.static_assert(key_cache_ptr.dtype.element_ty == tl.uint8)
    tl.static_assert(value_cache_ptr.dtype.element_ty == tl.bfloat16)
'''
    start=original.index('            # K : (HEAD_SIZE, TILE_SIZE)\n            if IS_3D and KV_QUANT_MODE == 0')
    end=original.index('            # V : (TILE_SIZE, HEAD_SIZE)\n',start)
    old_load=original[start:end]
    new_load='''            # K4 bytes hold even/odd D nibbles. Token scales stay after QK.
            packed_offset = (
                physical_block_idx[:, None] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + tl.arange(0, 64)[None, :]
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_k_cache_1
            )
            packed = tl.load(key_cache_ptr + packed_offset,
                mask=tile_mask[:, None], other=136, cache_modifier=KV_CACHE_MODIFIER)
            low = ((packed & 15).to(tl.int16) - 8).to(tl.bfloat16)
            high = ((packed >> 4).to(tl.int16) - 8).to(tl.bfloat16)
            K_load = tl.join(low, high).reshape(TILE_SIZE, 128).T
'''
    old_vscale='''            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )
'''
    old_pv='''        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)
'''
    new_pv='        acc += tl.dot(P.to(V.dtype), V)\n'
    rules=[(anchor,anchor+guard),(old_load,new_load),(old_vscale,''),(old_pv,new_pv)]
    text=original
    for old,new in rules:
        assert text.count(old)==1;text=text.replace(old,new)
    # Removing a fragment has no position for inverse replacement: keep its explicit anchor.
    inverse=text.replace(anchor+guard,anchor).replace(new_load,old_load).replace(new_pv,old_pv)
    marker='''                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
'''
    assert inverse.count(marker)==1
    inverse=inverse.replace(marker,marker+old_vscale)
    assert inverse==original
    before=ast.parse(original);after=ast.parse(text)
    defs=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    a,b=defs(before),defs(after)
    changed=[n for n in a if ast.dump(a[n])!=ast.dump(b[n])]
    assert changed==['kernel_unified_attention']
    assert ast.dump(a[changed[0]].args)==ast.dump(b[changed[0]].args)
    return text,dict(parent_sha256=PARENT_SHA,generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        changed_functions=changed,ABI_identical=True,wrapper_and_reducer_AST_identical=True,
        inverse_exact_source_proof=True,K4_unpack_and_K_token_scale_only=True,
        V_BF16_no_dequantization=True,supported='3D, D128, uint8 K4, BF16 V, mode2 scale ABI')

def load(parent,root):
    text,proof=render(parent);root=Path(root);root.mkdir(exist_ok=False)
    path=root/'generated_k4_reader.py';path.write_text(text)
    name='_K4_reader_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m)
    return m,dict(proof,generated_source=str(path))
