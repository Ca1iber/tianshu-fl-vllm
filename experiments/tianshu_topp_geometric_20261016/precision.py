# SPDX-License-Identifier: Apache-2.0
"""Geometric interval probes for standalone TopP, retaining exact semantic gates."""
import hashlib
import importlib.util
from pathlib import Path
import sys
SOURCE_SHA='64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
BASE_SHA='af8ffb90a6cbeee87c4769ec6710f03e6431da3e8c5fc47ea4a115e5be85adc8'
PARENT=Path(__file__).resolve().parent.parent/'tianshu_topp_actual_max_20261016/precision.py'
spec=importlib.util.spec_from_file_location('_tianshu_actualmax_generator',PARENT);previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
topk_ast=previous.topk_ast


def render_source(raw,mode):
    assert mode in ('max','geo')
    text=previous.render_source(raw,'max');assert hashlib.sha256(text.encode()).hexdigest()==BASE_SHA
    if mode=='max':return text
    start=text.index('        if TOPP_ENABLED and final_pivot == -float("inf"):');end=text.index('        # Sixth pass:',start)
    part=text[start:end]
    before='                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range\n'
    after='''                        arithmetic_middle = (max_range - min_range) * 0.5 + min_range
                        geometric_middle = tl.sqrt(tl.maximum(min_range, 0.0)) * tl.sqrt(tl.maximum(max_range, 0.0))
                        p_pivot_0 = tl.where(min_range > 0.0, geometric_middle, arithmetic_middle)
'''
    assert part.count(before)==2;part=part.replace(before,after);text=text[:start]+part+text[end:]
    assert topk_ast(text)==topk_ast(raw.decode())
    assert text[:start]+part.replace(after,before)+text[start+len(part):]==previous.render_source(raw,'max')
    return text


def load_kernel(native_source,directory):
    raw=Path(native_source).read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);kernels={};records={}
    for mode in ('max','geo'):
        source=directory/f'generated_{mode}.py';source.write_text(render_source(raw,mode))
        name=f'_tianshu_topp_geometric_{mode}';spec=importlib.util.spec_from_file_location(name,source)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        kernels[mode]=module._topk_topp_kernel;records[mode]=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),native_TopK_branch_AST_unchanged=True)
    return kernels,dict(native_source_sha256=SOURCE_SHA,base_sha256=BASE_SHA,modes=records,strict_CPU_tolerance_unchanged=1e-6,p_unchanged=True,RNG_unchanged=True,all_GPU_arithmetic_FP32=True,installed_source_modified=False,standalone_search_limit_unchanged=32,standalone_interval_tolerance_unchanged=1e-12)
