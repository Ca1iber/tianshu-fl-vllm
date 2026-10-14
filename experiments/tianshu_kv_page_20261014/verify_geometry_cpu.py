# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact guard and chained-loader checks; no vLLM/GPU imports."""
import argparse
import ast
import builtins
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile

from geometry_guard import generate, install, HYBRID_SHA


def fake_import(name, *args, **kwargs):
    assert name == 'vllm.v1.kv_cache_interface', name
    return SimpleNamespace(KVQuantMode=SimpleNamespace(NONE='none'))


def tensor(shape, dtype='bf16', device='cuda:0', cuda=True):
    return SimpleNamespace(shape=tuple(shape),ndim=len(shape),dtype=dtype,device=device,is_cuda=cuda)


def inputs(page):
    return dict(q=tensor((7,16,128)),k=tensor((9,page,2,128)),v=tensor((9,page,2,128)),
                out=tensor((7,16,128)),causal=True,window_size=(-1,-1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--hybrid-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raw=args.hybrid_source.read_bytes();assert hashlib.sha256(raw).hexdigest()==HYBRID_SHA
    original,candidate=generate(raw)
    assert candidate.replace('k.shape[1:] in ((16,2,128),(32,2,128))','k.shape[1:] == (16,2,128)')==original
    namespace=dict(torch=SimpleNamespace(bfloat16='bf16'),__builtins__=dict(vars(builtins),__import__=fake_import))
    old=dict(namespace);exec(compile(original,'<original-supported>','exec'),old)
    module=ModuleType('stub-hybrid');module.__dict__.update(namespace)
    wide,meta,_,_=install(module,raw);checked=0
    for page in (0,8,16,32,48,64):
        x=inputs(page);assert old['supported'](x)==(page==16)
        assert wide(x)==(page in (16,32));checked+=2
    invalids=[('q',tensor((7,15,128))),('q',tensor((7,16,64))),('q',tensor((7,16,128),'fp32')),
        ('k',tensor((9,32,2,128),'fp32')),('v',tensor((9,32,1,128))),
        ('out',tensor((7,16,128),'fp32')),('q',tensor((7,16,128),cuda=False)),
        ('v',tensor((9,32,2,128),device='cuda:1')),('causal',False),('window_size',(16,0)),
        ('softcap',1.),('output_scale',1.),('kv_quant_mode','fp8'),('alibi_slopes',1),
        ('qq_bias',1),('sinks',1),('mm_prefix_range',1),('use_td',True),('chunk_lookback',0)]
    for page in (16,32):
        for key,value in invalids:
            x=inputs(page)
            if key=='k':value=tensor((9,page,2,128),'fp32')
            if key=='v' and value.shape[-2]==1:value=tensor((9,page,1,128))
            elif key=='v':value=tensor((9,page,2,128),device='cuda:1')
            x[key]=value;assert not wide(x),(page,key);checked+=1
            if page==16:assert old['supported'](x)==wide(x);checked+=1
    source=Path(__file__).with_name('hooks')/'sitecustomize.py';tree=ast.parse(source.read_text())
    classes=ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef)],type_ignores=[])
    parent_calls=[];dispatch_sentinel=object()
    class ParentLoader(importlib.abc.Loader):
        def create_module(self,spec):parent_calls.append('create');return None
        def exec_module(self,module):
            parent_calls.append('exec');module.__dict__.update(namespace)
            module.__file__=str(args.hybrid_source.resolve());module.supported=old['supported']
            module.dispatch=dispatch_sentinel
    class NativePrefillFinder:
        def find_spec(self,fullname,path=None,target=None):
            parent_calls.append('find');return importlib.util.spec_from_loader(fullname,ParentLoader())
    with tempfile.TemporaryDirectory(prefix='tianshu-page-guard-',dir='/tmp') as tmp:
        env_key='TIANSHU_KV_PAGE_RECORD_DIR';previous=os.environ.get(env_key);os.environ[env_key]=tmp
        proxy=SimpleNamespace(meta_path=[])
        context=dict(importlib=importlib,hashlib=hashlib,json=json,os=os,Path=Path,sys=proxy,
            BASE=Path(__file__).resolve().parents[1],PROFILE='S8',expected='71b8368cca267314481f11e4efe5556b083ca5e86d6e80380ba4ab8632b54a19',__file__=str(source.resolve()))
        try:
            exec(compile(classes,str(source),'exec'),context)
            finder=context['Finder']();proxy.meta_path=[finder,NativePrefillFinder()]
            assert finder.find_spec('other.module') is None
            spec=finder.find_spec('vllm_fl.worker.iluvatar_hybrid_attention');assert spec.loader.create_module(spec) is None
            m=ModuleType(spec.name);spec.loader.exec_module(m)
            assert m.dispatch is dispatch_sentinel and parent_calls==['find','create','exec']
            assert m.supported(inputs(32));assert m.supported.__code__.co_filename=='<tianshu-page32-supported>'
            assert m.supported(inputs(16)) and not m.supported(inputs(64))
            witnesses=list(Path(tmp).glob('actual_geometry_*.json'));assert len(witnesses)==1
            assert json.loads(witnesses[0].read_text())['page_size']==32
        finally:
            if previous is None:os.environ.pop(env_key,None)
            else:os.environ[env_key]=previous
    result=dict(complete=True,CPU_only=True,GPU_kernel_and_service_not_tested=True,
        guard_checks=checked,exact_single_geometry_replacement=True,
        unsupported_geometry_dtype_device_quantization_options_rejected=True,
        existing_parent_loader_delegation_checked_with_CPU_stub=True,
        parent_dispatch_identity_preserved_in_stub=True,first_geometry_recorded_then_wrapper_removed=True,
        hybrid_source_sha256=HYBRID_SHA,predicate_provenance=meta,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
