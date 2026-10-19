# SPDX-License-Identifier: Apache-2.0
"""Actual metadata route and finder delegation tests; no GPU performance claim."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_grid_route_CPU',BASE/'grid_route.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Tensor:
    def __init__(self,batch=64,vocab=130560,dtype='torch.float32',device='cuda',stride=1):
        self.ndim=2;self.shape=(batch,vocab);self.dtype=dtype;self.device=SimpleNamespace(type=device);self.last_stride=stride
    def stride(self,dim):return self.last_stride if dim==1 else self.shape[1]


def check():
    records=[]
    def fixture(batch=64,attributes=None,kwargs=None,args_empty=False,expected=False):
        kw=dict(BATCH_SIZE=batch,VOCAB_SIZE=130560,TOPK_ENABLED=False,TOPP_ENABLED=True,BLOCK_SIZE=8192,BLOCK_SIZE_TRUNC=4096)
        kw.update(kwargs or {});args=() if args_empty else (Tensor(batch,**(attributes or {})),130560,object())
        actual=module.supported(args,kw);assert actual==expected,(batch,attributes,kw,actual)
        records.append(dict(batch=batch,attributes=attributes,kwargs=kw,empty=args_empty,supported=actual))
    for batch in (0,1,7,8,16,24,31,32,33,40,48,56,63,64,65,128):fixture(batch,expected=32<=batch<=64)
    for kw in ({'TOPK_ENABLED':True},{'TOPP_ENABLED':False},{'VOCAB_SIZE':130559},
            {'BLOCK_SIZE':16384},{'BLOCK_SIZE_TRUNC':8192},{'BATCH_SIZE':63}):fixture(kwargs=kw)
    for attr in ({'dtype':'torch.float16'},{'dtype':'torch.bfloat16'},{'device':'cpu'},
                 {'stride':2},{'vocab':130559}):fixture(attributes=attr)
    fixture(args_empty=True);fixture(batch=32,expected=True);fixture(batch=63,expected=True)
    assert len(records)==30
    # Execute the actual finder class without running sitecustomize imports.
    import ast,importlib.abc,sys,textwrap
    source=(BASE/'hooks/sitecustomize.py').read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Finder')
    body='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]);ns=dict(importlib=__import__('importlib'),sys=sys,Loader=lambda loader:('wrapped',loader));exec(compile(body,'<actual_grid_finder>','exec'),ns)
    finder=ns['Finder']();old=sys.meta_path
    class Delegate:
        def __init__(self,fail=False):self.calls=0;self.fail=fail
        def find_spec(self,*args):
            self.calls+=1;assert finder not in sys.meta_path
            if self.fail:raise RuntimeError('CPU fixture failure')
            return SimpleNamespace(loader='original')
    try:
        d=Delegate();sys.meta_path=[finder,d]
        assert finder.find_spec('other.module') is None and sys.meta_path[0] is finder and d.calls==0
        assert finder.find_spec('vllm.v1.sample.ops.topk_topp_triton').loader==('wrapped','original') and d.calls==1 and sys.meta_path[0] is finder
        d=Delegate(True);sys.meta_path=[finder,d]
        try:finder.find_spec('vllm.v1.sample.ops.topk_topp_triton')
        except RuntimeError:pass
        else:raise AssertionError('error not propagated')
        assert sys.meta_path[0] is finder and d.calls==1
    finally:sys.meta_path=old
    spec=importlib.util.spec_from_file_location('_grid_scratch_CPU',BASE/'kernel_route.py')
    kernel_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel_module)
    class Storage:
        def __init__(self,shape):self.shape=shape;self.device='CPU_FIXTURE';self.dtype='FP32_FIXTURE'
        def new_empty(self,shape):return Storage(shape)
    class Kernel:
        fn=SimpleNamespace(__module__='CPU_FIXTURE')
        def __getitem__(self,grid):
            def launch(*args,**kw):
                assert args[2].shape[0]>=grid[0]
                return SimpleNamespace(metadata=SimpleNamespace(num_warps=kw['num_warps'],num_stages=kw['num_stages'],shared=0),n_regs=None,n_spills=None)
            return launch
    count=0
    for batch in (8,16,17,32,63,64):
        for topk in (False,True):
            for plan in kernel_module.PLANS:
                original=min(16,batch);route=kernel_module.Router(Kernel(),plan,corrected_kernel={'geo':Kernel()})
                route[(original,)](Storage((batch,130560)),130560,Storage((original,130560)),None,None,None,None,
                    BATCH_SIZE=batch,VOCAB_SIZE=130560,TOPK_ENABLED=topk,TOPP_ENABLED=True)
                expected=min(batch,original*plan[4]) if plan[0].startswith('fixed') and not topk else original
                assert route.metadata['grid']==[expected] and route.metadata['actual_buffer_rows']>=expected
                if expected>original:assert route.metadata['actual_buffer_rows']==64
                count+=1
    assert count==96 and len(kernel_module.SCRATCH)==1
    hashes={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ('grid_route.py','kernel_route.py','hooks/sitecustomize.py','check_cpu.py')}
    result=dict(complete=True,CPU_only=True,GPU_test=False,routing_cases=len(records),import_delegation_cases=3,grid_buffer_cases=96,shared_scratch_buffers=1,sources=hashes,records=records)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','records')}))


if __name__=='__main__':check()
