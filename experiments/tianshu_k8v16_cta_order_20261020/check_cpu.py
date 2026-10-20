# SPDX-License-Identifier: Apache-2.0
"""CPU-only bijection plus actual qualified launch classes with fake kernels."""
import hashlib,itertools,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import source
import launch
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_k8v16_address32_20261020/qualified_S16_reader.py'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class FakeKernel:
    def __init__(self):self.calls=[]
    def __getitem__(self,grid):
        def run(*args,**kw):
            self.calls.append((tuple(grid),kw))
            return SimpleNamespace(metadata=SimpleNamespace(num_warps=kw.get('num_warps',1),num_stages=kw.get('num_stages',1),shared=0),n_regs=0,n_spills=0,hash='CPU_fake_not_a_compiled_GPU_kernel')
        return run

def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    proofs={};geometry=[];calls=[]
    for mode in source.ORDERS:
        text,proof=source.render(PARENT,mode);(root/f'{mode}.py').write_text(text);proofs[mode]=proof
    for n in (1,2,4,8,16,24,32,40,41,48,56,63,64):
      qblocks=n//2+n
      for segments in (4,8,16,32,64):
        old=(qblocks,2,segments)
        expected=set(itertools.product(range(old[0]),range(old[1]),range(old[2])))
        for mode,order in source.ORDERS.items():
          new=source.remap_grid(old,mode);observed=[]
          for z,y,x in itertools.product(range(new[2]),range(new[1]),range(new[0])):
            coords=(x,y,z);observed.append(tuple(coords[order.index(i)] for i in range(3)))
          assert len(observed)==len(set(observed))==len(expected) and set(observed)==expected
          assert source.remap_grid((qblocks,2),mode,False)==(qblocks,2)
          geometry.append(dict(n=n,segments=segments,mode=mode,old=list(old),new=list(new),CTAs=len(observed),bijection=True))
    for n in (1,8,32,40,41,56,64):
      for length in (4096,16384):
        cfg=launch.native.choose_config(n,length);segments=cfg['segments'];grid=(n//2+n,2,segments)
        native_args=None
        for mode in source.ORDERS:
          fake=FakeKernel();wrapper=launch.PermutedKernel(fake,mode);producer=launch.native.Producer(wrapper)
          producer.pure_decode=True;producer.decode_config=cfg
          kwargs=dict(BLOCK_SIZE=16,HEAD_SIZE=128,num_query_heads=16,num_queries_per_kv=8,KV_QUANT_MODE=0,
            USE_TD=False,USE_TD_QO=False,query_ptr=torch.empty(0,dtype=torch.bfloat16),key_cache_ptr=torch.empty(0,dtype=torch.uint8),
            value_cache_ptr=torch.empty(0,dtype=torch.bfloat16),IS_3D=True,num_seqs=n,BLOCK_M=16,BLOCK_Q=2,NUM_SEGMENTS_PER_SEQ=segments)
          producer[grid](**kwargs);actual,emitted=fake.calls[-1]
          assert actual==source.remap_grid(grid,mode)
          comparable={k:v for k,v in emitted.items() if not k.endswith('_ptr')}
          if native_args is None:native_args=comparable
          else:assert comparable==native_args
          assert wrapper.last['num_warps']==cfg['num_warps'] and wrapper.last['num_stages']==1
          # 2D producer/fallback retains its own original dimensions.
          simple=FakeKernel();wrapped2=launch.PermutedKernel(simple,mode)
          wrapped2[(n//2+n,2)](IS_3D=False)
          assert simple.calls[-1][0]==(n//2+n,2)
          calls.append(dict(n=n,context=length,mode=mode,original=list(grid),actual=list(actual),launch_arguments_unchanged=True))
    try:source.render(PARENT,'unknown')
    except AssertionError:pass
    else:raise AssertionError('invalid mode accepted')
    altered=root/'altered_reader.py';altered.write_bytes(PARENT.read_bytes()+b'\n')
    try:source.render(altered,'heads_first')
    except AssertionError:pass
    else:raise AssertionError('unqualified source accepted')
    paths=[BASE/'source.py',BASE/'launch.py',Path(__file__),PARENT,launch.NATIVE_PATH,
        launch.NATIVE_PATH.with_name('guard.py'),launch.native.LEGACY,launch.native.POLICY]
    result=dict(complete=True,CPU_only=True,GPU_not_executed=True,service_or_quality_not_run=True,
        geometry_cases=len(geometry),actual_native_Producer_cases=len(calls),all_2D_fallbacks_preserved=True,
        all_CTA_coordinate_sets_equal=True,launch_source_proof=launch.native.source_proof(),
        provenance=proofs,geometry=geometry,launch_calls=calls,sources={str(p):sha(p) for p in paths})
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('geometry','launch_calls','provenance','sources')}))
if __name__=='__main__':main()
