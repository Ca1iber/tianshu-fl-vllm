# SPDX-License-Identifier: Apache-2.0
"""Pure CPU weight-only fitting; no model/evaluation data or GPU import."""
import importlib.util
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
EXP=BASE.parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
reader=load('_qkvo64_original_quant_reader',EXP/'tianshu_W4_attention64_20261024/common.py')
fit_O=load('_qkvo64_O_weight_fitter',EXP/'tianshu_W4_affine_fit_CPU_20261024/study_cpu.py')
fit_QKV=load('_qkvo64_QKV_weight_fitter',EXP/'tianshu_W4_QKV_affine_fit_CPU_20261024/study_cpu.py')
ahash=reader.ahash
unpack=reader.unpack

def quant(words,mode):
    assert mode=='affine' and words.dtype.name=='uint16'
    assert words.shape in ((2048,2048),(2560,2048))
    fitter=fit_O if words.shape[0]==2048 else fit_QKV
    packed,scale,zero,initial,error,history=fitter.fitted(words)
    assert error<=initial and len(history)==6
    return unpack(packed),scale,zero,packed
