# SPDX-License-Identifier: Apache-2.0
"""Independent PV-only variant; native QK, softmax, writer and reducer preserved."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_k8vi8_heads_v2_20261021'
spec=importlib.util.spec_from_file_location('_PV_int8_native_scaled_V_parent',PARENT/'source.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
SOURCE,SOURCE_SHA=parent.SOURCE,parent.SOURCE_SHA

def render(raw):
    original,parent_proof=parent.render(raw)
    edits=[('from vllm.triton_utils import tl, triton\n','from vllm.triton_utils import tl, triton\nfrom _tianshu_k8vi8_pv_integer import quantized_pv\n'),
        ('    KV_QUANT_MODE: tl.constexpr = 0,\n','    KV_QUANT_MODE: tl.constexpr = 0,\n    PV_INT8: tl.constexpr = False,\n'),
        ('    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2\n','    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2\n    tl.static_assert(not PV_INT8 or (IS_3D and KV_QUANT_MODE == 2 and not SLIDING_WINDOW and value_cache_ptr.dtype.element_ty == tl.int8))\n'),
        ('        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)\n','        if PV_INT8:\n            V = V_load\n        else:\n            V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)\n'),
        ('        if USE_PER_TOKEN_HEAD_SCALES:\n            # Per-token-head quant: apply v_scale to P instead of V.\n','        if PV_INT8:\n            acc += quantized_pv(P, V, v_token_head_scales)\n        elif USE_PER_TOKEN_HEAD_SCALES:\n            # Per-token-head quant: apply v_scale to P instead of V.\n')]
    text=original
    for before,after in edits:assert text.count(before)==1,(before,text.count(before));text=text.replace(before,after)
    restored=text
    for before,after in reversed(edits):assert restored.count(after)==1;restored=restored.replace(after,before)
    assert restored==original
    def fn(source,name):return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('reduce_segments','unified_attention','_cast_kv_tile'):
        assert ast.dump(fn(text,name))==ast.dump(fn(original,name))
    return text,dict(parent=parent_proof,generated_sha256=hashlib.sha256(text.encode()).hexdigest(),all_edits_reversed_to_exact_parent=True,
        native_QK_and_softmax_code_unchanged=True,public_API_and_reducer_AST_unchanged=True,default_flag_preserves_native_V8_path=True,
        new_path_requires_3D_INT8_V_without_sliding_window=True,writer_and_cache_layout_not_modified=True,
        new_value_and_probability_approximation_requires_fresh_Math=True)

def load_modules(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    helper_name='_tianshu_k8vi8_pv_integer';spec=importlib.util.spec_from_file_location(helper_name,BASE/'kernel_pv.py')
    helper=importlib.util.module_from_spec(spec);sys.modules[helper_name]=helper;spec.loader.exec_module(helper)
    text,proof=render(SOURCE.read_bytes());path=root/'generated_reader.py';path.write_text(text)
    name='_tianshu_k8vi8_pv_int8_reader';spec=importlib.util.spec_from_file_location(name,path)
    reader=importlib.util.module_from_spec(spec);sys.modules[name]=reader;spec.loader.exec_module(reader)
    return reader,helper,dict(proof,generated_source=str(path))
