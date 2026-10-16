# SPDX-License-Identifier: Apache-2.0
"""Adapt only the initial outlier cutoff; keep exact fallback/search gates."""
import hashlib
import importlib.util
from pathlib import Path
import sys

PARENT=Path(__file__).resolve().parents[1]/'tianshu_topp_geometric_20261016/precision.py'
spec=importlib.util.spec_from_file_location('_tianshu_geometric_generator',PARENT)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
topk_ast=previous.topk_ast
SOURCE_SHA='64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
BASE_SHA='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
MODES=('geo','fixed_bias30','fixed_bias25','fixed_bias20')


def render_source(raw,mode):
    assert mode in MODES
    base=previous.render_source(raw,'geo');assert hashlib.sha256(base.encode()).hexdigest()==BASE_SHA
    if mode=='geo':return base
    bias={'fixed_bias30':3.0,'fixed_bias25':2.5,'fixed_bias20':2.0}[mode]
    start=base.index('        if TOPP_ENABLED and final_pivot == -float("inf"):')
    end=base.index('        # Sixth pass:',start)
    part=base[start:end]
    before='                sigma = sigma + tl.abs(sigma) * -0.25\n'
    after=before+f'                sigma = tl.maximum(sigma, std_logit - {bias})\n'
    assert part.count(before)==1
    part=part.replace(before,after)
    text=base[:start]+part+base[end:]
    assert text[:start]+part.replace(after,before)+text[start+len(part):]==base
    assert topk_ast(text)==topk_ast(raw.decode())
    return text


def load_kernel(native_source,directory):
    raw=Path(native_source).read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);kernels={};records={}
    for mode in MODES:
        path=directory/f'generated_{mode}.py';path.write_text(render_source(raw,mode))
        name='_tianshu_topp_adaptive_'+mode;spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        kernels[mode]=module._topk_topp_kernel
        records[mode]=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),native_TopK_branch_AST_unchanged=True)
    return kernels,dict(native_source_sha256=SOURCE_SHA,base_sha256=BASE_SHA,modes=records,
        strict_CPU_tolerance_unchanged=1e-6,p_unchanged=True,RNG_unchanged=True,all_GPU_arithmetic_FP32=True,
        installed_source_modified=False,standalone_search_limit_unchanged=32,standalone_interval_tolerance_unchanged=1e-12,
        only_initial_outlier_cutoff_changed=True,full_range_fallback_retained=True)
