# SPDX-License-Identifier: Apache-2.0
"""Current full-attention reader with strict/row-recip P-quant helpers kept distinct."""
import hashlib,importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_INT8_PV_attention_20261024'
spec=importlib.util.spec_from_file_location('_strict_PV_source_parent',PARENT/'codec.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
READER,WRITER,LAYOUT,REFERENCE,HELPER,PRIMITIVE_GATE,WRITER_GATE=(
    getattr(parent,name) for name in ('READER','WRITER','LAYOUT','REFERENCE','HELPER','PRIMITIVE_GATE','WRITER_GATE'))
FAST_HELPER=BASE/'kernel_pv_recip.py'
SOURCE_SHA=parent.SOURCE_SHA
HELPER_MODULES={'strict':'_tianshu_k8vi8_pv_integer','row_recip':'_tianshu_k8vi8_pv_row_recip'}

def render_reader(raw,variant):
    assert variant in HELPER_MODULES
    original,proof=parent.render_reader(raw)
    before='from _tianshu_k8vi8_pv_integer import quantized_pv'
    after=f'from {HELPER_MODULES[variant]} import quantized_pv'
    assert original.count(before)==1
    generated=original.replace(before,after)
    assert generated.replace(after,before)==original
    return generated,dict(parent=proof,variant=variant,
        helper_import_only_changed=True,generated_sha256=hashlib.sha256(generated.encode()).hexdigest())

def load_helper():
    modules={}
    for variant,path in [('strict',HELPER),('row_recip',FAST_HELPER)]:
        name=HELPER_MODULES[variant]
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        modules[variant]=module
    return modules
