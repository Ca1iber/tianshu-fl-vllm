# SPDX-License-Identifier: Apache-2.0
"""Build the actual small-arena subclass from original source without a GPU."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import torch
from worker_hook import check_dense
BASE=Path(__file__).resolve().parent
PIPE=BASE.parent/'tianshu_k8v16_pipeline_20261019'
WRAPPER=BASE.parent/'tianshu_small_arena_20261020/policy.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(root):
    fake=SimpleNamespace(full=torch.full,arange=torch.arange,bfloat16=torch.bfloat16,int32=torch.int32,
        cuda=SimpleNamespace(is_current_stream_capturing=lambda:False,memory_allocated=lambda:0))
    dense_module=ModuleType('_dense_prefill_original_CPU_guard');dense_module.__file__=str(PIPE/'dense_prefill.py');sys.modules[dense_module.__name__]=dense_module
    dense_module.__dict__.update(torch=fake)
    nodes=[n for n in ast.parse((PIPE/'dense_prefill.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('DensePrefill','shape_for')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PIPE/'dense_prefill.py'),'exec'),dense_module.__dict__)
    native=ModuleType('_bounded_prefill_original_CPU_guard');native.__file__=str(PIPE/'bounded_prefill.py');sys.modules[native.__name__]=native
    native.__dict__.update(torch=fake,DensePrefill=dense_module.DensePrefill,shape_for=dense_module.shape_for,CAPACITY_PAGES=8192)
    nodes=[n for n in ast.parse((PIPE/'bounded_prefill.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PIPE/'bounded_prefill.py'),'exec'),native.__dict__)
    original=native.BoundedDensePrefill
    spec=importlib.util.spec_from_file_location('_actual_small_arena_CPU_guard',WRAPPER);small=importlib.util.module_from_spec(spec);sys.modules[spec.name]=small;spec.loader.exec_module(small)
    row=small.install(native,'candidate',root/'small_arena_CPU_guard')
    actual=native.BoundedDensePrefill();guard=check_dense(actual)
    assert native.BoundedDensePrefill.__call__ is original.__call__
    control=original();control.reserve('cpu');actual.reserve('cpu')
    assert torch.equal(control.storage,actual.storage) and torch.equal(control.dense_ids,actual.dense_ids)
    assert control.storage.shape==actual.storage.shape==(8193,2,16,2,128)
    assert control.reservation==actual.reservation
    rejected=0
    for item in (original(),native.BoundedDensePrefill()):
        if type(item) is not original:item.limit_pages=4096
        try:check_dense(item)
        except AssertionError:rejected+=1
        else:raise AssertionError('unsupported dense instance passed')
    assert rejected==2 and not torch.cuda.is_initialized()
    return dict(complete=True,actual_native_class_AST_and_small_policy_executed=True,
        actual_inherited_call_and_reserve_layout_verified=True,CPU_real_storage_elements=control.storage.numel()+actual.storage.numel(),
        no_GPU_initialized=True,unsupported_native_or_wrong_capacity_rejected=rejected,guard=guard,
        sources={str(p):sha(p) for p in (Path(__file__).resolve(),BASE/'worker_hook.py',PIPE/'dense_prefill.py',PIPE/'bounded_prefill.py',WRAPPER)})
