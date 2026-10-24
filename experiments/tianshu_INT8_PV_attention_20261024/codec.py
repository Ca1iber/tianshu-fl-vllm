# SPDX-License-Identifier: Apache-2.0
"""Current qualified K8 reader plus native dynamic INT8 V and optional integer PV."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
READER=BASE.parent/'tianshu_k8v16_address32_20261020/qualified_S16_reader.py'
WRITER=BASE.parent/'tianshu_k8vi8_heads_v2_20261021/writer.py'
LAYOUT=WRITER.with_name('layout.py')
REFERENCE=WRITER.with_name('reference_quantize.py')
HELPER=BASE.parent/'tianshu_k8vi8_pv_int8_20261021/kernel_pv.py'
PRIMITIVE_GATE=HELPER.with_name('PRIMITIVE_GPU_GATE_32g.json')
WRITER_GATE=WRITER.with_name('GPU_GATE_32g.json')
SOURCE_SHA='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'

def render_reader(raw,mode=None):
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    original=raw.decode()
    edits=[
      ('from vllm.triton_utils import tl, triton\n','from vllm.triton_utils import tl, triton\nfrom _tianshu_k8vi8_pv_integer import quantized_pv\n'),
      ('    KV_QUANT_MODE: tl.constexpr = 0,\n','    KV_QUANT_MODE: tl.constexpr = 0,\n    PV_INT8: tl.constexpr = False,\n'),
      ('    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2\n',
       '    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2\n    tl.static_assert(not PV_INT8 or (IS_3D and KV_QUANT_MODE == 2 and not SLIDING_WINDOW and not USE_TD and value_cache_ptr.dtype.element_ty == tl.int8))\n'),
      ('if KV_QUANT_MODE == 0 and data.dtype == tl.uint8:','if (KV_QUANT_MODE == 0 or KV_QUANT_MODE == 2) and data.dtype == tl.uint8:'),
      ('        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)\n',
       '        if PV_INT8:\n            V = V_load\n        else:\n            V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)\n'),
      ('        if USE_PER_TOKEN_HEAD_SCALES:\n            # Per-token-head quant: apply v_scale to P instead of V.\n',
       '        if PV_INT8:\n            acc += quantized_pv(P, V, v_token_head_scales)\n        elif USE_PER_TOKEN_HEAD_SCALES:\n            # Per-token-head quant: apply v_scale to P instead of V.\n'),
      ('        physical_block_idx = tl.load(\n            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE\n        ).to(tl.int64)',
       '        if KV_QUANT_MODE == 2:\n            physical_block_idx = tl.load(\n                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE,\n                mask=tile_mask, other=0,\n            ).to(tl.int64)\n        else:\n            physical_block_idx = tl.load(\n                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE\n            ).to(tl.int64)')]
    generated=original
    for before,after in edits:
        assert generated.count(before)==1,(before,generated.count(before))
        generated=generated.replace(before,after)
    restored=generated
    for before,after in reversed(edits):
        assert restored.count(after)==1
        restored=restored.replace(after,before)
    assert restored==original
    def function(text,name):
        return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('reduce_segments','unified_attention'):
        assert ast.dump(function(generated,name))==ast.dump(function(original,name))
    return generated,dict(parent_sha256=SOURCE_SHA,generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        all_edits_inverse_restore_exact_parent=True,native_QK_and_online_softmax_unchanged=True,
        public_API_and_reducer_AST_unchanged=True,current_unpaired_K_load_preserved=True,
        page_table_tails_masked_for_both_dynamic_V8_paths=True,requires_fresh_model_quality=True)

def load_helper():
    name='_tianshu_k8vi8_pv_integer'
    spec=importlib.util.spec_from_file_location(name,HELPER)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
