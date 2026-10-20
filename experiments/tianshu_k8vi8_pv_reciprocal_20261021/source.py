# SPDX-License-Identifier: Apache-2.0
"""One constexpr switch in the frozen full attention variant, unchanged layout."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_k8vi8_pv_int8_20261021'
spec=importlib.util.spec_from_file_location('_PV_reciprocal_frozen_source',PARENT/'source.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
SOURCE,SOURCE_SHA=parent.SOURCE,parent.SOURCE_SHA

def render(raw):
    original,proof=parent.render(raw)
    edits=[('    PV_INT8: tl.constexpr = False,\n','    PV_INT8: tl.constexpr = False,\n    PV_RECIPROCAL: tl.constexpr = False,\n'),
        ('quantized_pv(P, V, v_token_head_scales)','quantized_pv(P, V, v_token_head_scales, PV_RECIPROCAL)')]
    text=original
    for before,after in edits:assert text.count(before)==1;text=text.replace(before,after)
    restored=text
    for before,after in reversed(edits):assert restored.count(after)==1;restored=restored.replace(after,before)
    assert restored==original
    def function(t,n):return next(f for f in ast.parse(t).body if isinstance(f,ast.FunctionDef) and f.name==n)
    for name in ('reduce_segments','unified_attention','_cast_kv_tile'):assert ast.dump(function(text,name))==ast.dump(function(original,name))
    return text,dict(parent=proof,all_edits_reversible_to_exact_frozen_parent=True,reciprocal_switch_default=False,
        native_QK_softmax_reducer_public_API_unchanged=True,new_P_rounding_requires_new_numeric_checks=True,
        generated_sha256=hashlib.sha256(text.encode()).hexdigest())

def load_modules(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    name='_tianshu_k8vi8_pv_integer';spec=importlib.util.spec_from_file_location(name,BASE/'kernel_pv.py')
    helper=importlib.util.module_from_spec(spec);sys.modules[name]=helper;spec.loader.exec_module(helper)
    text,proof=render(SOURCE.read_bytes());path=root/'generated_reader.py';path.write_text(text)
    name='_tianshu_k8vi8_pv_reciprocal_reader';spec=importlib.util.spec_from_file_location(name,path)
    reader=importlib.util.module_from_spec(spec);sys.modules[name]=reader;spec.loader.exec_module(reader)
    return reader,helper,dict(proof,generated_source=str(path))
