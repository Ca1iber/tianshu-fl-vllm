# SPDX-License-Identifier: Apache-2.0
"""CPU-only real call/fallback and allocation-contract boundary checks, no torch import."""
import argparse,ast,hashlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
BASE=Path(__file__).resolve().parent;PARENT=BASE.parent/'tianshu_k8v16_pipeline_20261019'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('_small_arena_cpu_policy',BASE/'policy.py');policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve()
assert root.is_relative_to('/workspace/benchmark-results');root.mkdir(parents=True,exist_ok=False)
class Tensor:
    def __init__(self,shape,width):self.shape=shape;self.width=width
    def numel(self):
        count=1
        for size in self.shape:count*=size
        return count
    def element_size(self):return self.width
class FakeTorch:
    bfloat16='BF16';int32='int32'
    def __init__(self):self.allocated=0;self.calls=[];self.cuda=SimpleNamespace(is_current_stream_capturing=lambda:False,memory_allocated=lambda:self.allocated)
    def full(self,shape,value,device,dtype):
        assert value==17. and device=='cuda' and dtype==self.bfloat16
        result=Tensor(shape,2);self.allocated+=result.numel()*2;self.calls.append(('full',shape));return result
    def arange(self,n,device,dtype):
        assert device=='cuda' and dtype==self.int32
        result=Tensor((n,),4);self.allocated+=n*4;self.calls.append(('arange',n));return result
class Dense:
    def __init__(self):self.records={}
    def verify_last(self,*args):raise AssertionError('CPU does not execute GPU reference')
source=PARENT/'bounded_prefill.py';tree=ast.parse(source.read_text())
body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.Assign))]
shape_node=next(n for n in ast.parse((PARENT/'dense_prefill.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='shape_for')
rows=[]
for mode,pages in policy.CAPACITIES.items():
    fake=FakeTorch();namespace=dict(torch=fake,DensePrefill=Dense,__name__='_capacity_cpu_'+mode)
    exec(compile(ast.Module(body=[shape_node]+body,type_ignores=[]),str(source),'exec'),namespace)
    module=SimpleNamespace(__file__=str(source),**namespace)
    proof=policy.install(module,mode,root/mode);instance=module.BoundedDensePrefill()
    for n,length in [(1,131072),(1,131073),(2,65536),(2,65537),(8,16384),(8,16385),(64,2048),(64,2049),(1,524288),(1,524289)]:
        expected=n*((length+15)//16)<=pages
        assert namespace['fits'](n,length,instance.limit_pages)==expected
        if not expected:
            used=SimpleNamespace(numel=lambda n=n:n)
            assert instance(dict(k=None,v=None,block_table=None,seqused_k=used,max_seqlen_k=length)) is None
            assert instance.storage is None and fake.calls==[]
        rows.append(dict(mode=mode,sequences=n,length=length,fits=expected,actual_native_fallback_checked=not expected))
    instance.reserve();storage,ids=instance.storage,instance.dense_ids;allocated=fake.allocated
    instance.reserve();assert instance.storage is storage and instance.dense_ids is ids and fake.allocated==allocated and len(fake.calls)==2
    assert instance.reservation['scratch_bytes']==(pages+1)*16384 and instance.reservation['table_bytes']==pages*4
    assert instance.reservation['allocated_after']-instance.reservation['allocated_before']==allocated
sources=[BASE/'policy.py',Path(__file__),source,PARENT/'dense_prefill.py',PARENT/'routing.py']
result=dict(complete=True,CPU_only=True,mode_capacities=policy.CAPACITIES,capacity_boundary_checks=rows,
            original_GPU_call_and_fallback_inherited=True,mocked_reservation_bytes_and_reuse_checked=True,
            no_torch_or_GPU_imported=True,sources={str(p):sha(p) for p in sources})
(root/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
for p in sources:
    out=root/'source_snapshot'/p.resolve().relative_to('/');out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(p.read_bytes())
print(json.dumps(dict(complete=True,CPU_only=True,boundary_checks=len(rows))))
