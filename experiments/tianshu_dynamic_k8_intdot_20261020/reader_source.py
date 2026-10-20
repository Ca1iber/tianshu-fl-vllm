# SPDX-License-Identifier: Apache-2.0
"""Private reader: dynamic K metadata and INT8 QK only for 3D decode."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
PARENT_SHA='dc69f525d8c7686a8977f50e901af39820ce92486894890631c53fc01dc46700'
PATCHES=(
 ('from vllm.triton_utils import tl, triton\n','from vllm.triton_utils import tl, triton\nfrom triton.language.extra.corex import libdevice as dynamic_k_libdevice\n'),
 ('    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1 and Q_IS_FP8\n',
  '    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1 and Q_IS_FP8\n    USE_DYNAMIC_K: tl.constexpr = KV_QUANT_MODE == 0 and key_cache_ptr.dtype.element_ty == tl.uint8\n    USE_INT8_QK: tl.constexpr = IS_3D and USE_DYNAMIC_K\n'),
 ('    # iterate through tiles (now limited to the sliding window range)\n',
  '''    if USE_INT8_QK:
        q_maximum = tl.maximum(tl.max(tl.abs(Q.to(tl.float32)), axis=1), 1.e-8)
        q_row_scale = tl.div_rn(q_maximum, 127.)
        q_rounded = dynamic_k_libdevice.nearbyint(tl.div_rn(Q.to(tl.float32), q_row_scale[:, None]))
        Q_int8 = tl.minimum(tl.maximum(q_rounded, -127.), 127.).to(tl.int8)

    # iterate through tiles (now limited to the sliding window range)
'''),
 ('''        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)''',
  '''        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE,
            mask=tile_mask, other=0,
        ).to(tl.int64)'''),
 ('''        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)''',
  '''        if USE_DYNAMIC_K:
            scale_addresses = (key_cache_ptr + physical_block_idx * stride_k_cache_0
                + 12288 + (seq_offset % BLOCK_SIZE) * 8 + kv_head_idx * 4).to(tl.pointer_type(tl.float32))
            token_k_scale = tl.load(scale_addresses, mask=tile_mask, other=1., cache_modifier=KV_CACHE_MODIFIER if IS_3D else "")
            signed_keys = K_load.to(tl.int8, bitcast=True)
            if USE_INT8_QK:
                K = signed_keys
            else:
                K = (signed_keys.to(tl.float32) * token_k_scale[None, :]).to(Q.dtype)
        else:
            K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)'''),
 ('''        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:''',
  '''        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_INT8_QK:
            integer_scores = tl.dot(Q_int8, K, out_dtype=tl.int32)
            S += ((integer_scores.to(tl.float32) * q_row_scale[:, None]) * token_k_scale[None, :]) * score_scale
        elif USE_PER_TOKEN_HEAD_SCALES:'''),
)

def generate(original):
    assert hashlib.sha256(original.encode()).hexdigest()==PARENT_SHA
    candidate=original
    for before,after in PATCHES:
        assert candidate.count(before)==1
        candidate=candidate.replace(before,after)
    reverse=candidate
    for before,after in reversed(PATCHES):
        assert reverse.count(after)==1
        reverse=reverse.replace(after,before)
    assert reverse==original
    def functions(text):return {x.name:x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
    a,b=functions(original),functions(candidate)
    assert a.keys()==b.keys()
    assert {name for name in a if ast.dump(a[name])!=ast.dump(b[name])}=={'kernel_unified_attention'}
    assert ast.dump(a['kernel_unified_attention'].args)==ast.dump(b['kernel_unified_attention'].args)
    return candidate

def load(parent,output):
    parent=Path(parent);output=Path(output)
    generated=generate(parent.read_text());output.mkdir(parents=True,exist_ok=False)
    p=output/'generated_dynamic_k8_intdot_reader.py';p.write_text(generated)
    name='_tianshu_dynamic_k8_intdot_'+hashlib.sha256(str(p).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(parent_source=str(parent),parent_sha256=PARENT_SHA,generated_source=str(p),
                       generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                       argument_ABI_unchanged=True,exact_reversal_verified=True,helper_and_reducer_AST_unchanged=True,
                       physical_page_bytes=12416,K_scales_offset_bytes=12288,query_quantization_3D_only=True,
                       table_loads_masked=True,model_or_quality_not_verified=True)
