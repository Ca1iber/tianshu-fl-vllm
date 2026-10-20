# SPDX-License-Identifier: Apache-2.0
"""Check exact profile parity and execute routing with non-GPU fake kernels."""
import hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import candidate_profile,policy
BASE=Path(__file__).resolve().parent

def main():
    root=Path('/workspace/benchmark-results/20261020_pair_service_PROFILE_CPU_v2_tianshu');root.mkdir(exist_ok=False)
    states={mode:candidate_profile.prepare(root/mode,mode) for mode in ('control','candidate')}
    a,b=states.values();assert a[0]==b[0] and a[3]==b[3]
    for key,value in a[2].items():
        other=b[2][key]
        if key=='TIANSHU_PAIR_LOAD_MODE':assert (value,other)==('control','candidate')
        else:assert value.replace(str(root/'control'),'<RUN>')==other.replace(str(root/'candidate'),'<RUN>'),key
    class Fake:
        def __init__(self):self.fn=lambda:None;self.calls=[]
        def __getitem__(self,grid):
            def run(**kw):self.calls.append((grid,dict(kw)));return SimpleNamespace(metadata=SimpleNamespace(num_warps=1,num_stages=1),n_regs=1,n_spills=0,hash='fake')
            return run
    class Cache:
        dtype='torch.uint8'
        def data_ptr(self):return 4096
        def stride(self,dim=None):return (12288,256,128,1) if dim is None else (12288,256,128,1)[dim]
    n=0
    for mode in ('control','candidate'):
        for is3d in (False,True):
            control,candidate=Fake(),Fake();record={'launches':{}}
            wrapper=policy.RoutedKernel(control,candidate,mode,{'active_graph':None},lambda:False,lambda:None,record)
            kw=dict(IS_3D=is3d,num_seqs=64,NUM_SEGMENTS_PER_SEQ=4,TILE_SIZE=16,BLOCK_SIZE=16,HEAD_SIZE=128,HEAD_SIZE_PADDED=128,
                KV_QUANT_MODE=0,USE_TD=False,USE_TD_QO=False,key_cache_ptr=Cache(),value_cache_ptr=SimpleNamespace(dtype='torch.bfloat16'))
            wrapper[(96,2,4)](**kw)
            selected=mode=='candidate' and is3d
            assert len(candidate.calls)==int(selected) and len(control.calls)==int(not selected)
            assert (candidate if selected else control).calls==[((96,2,4),kw)]
            assert next(iter(record['launches'].values()))['selected_pair_rows']==selected;n+=1
    gate=dict(complete=True,CPU_only=True,GPU_service_not_executed=True,frozen_source_count=len(a[3]),environment_count=len(a[2]),
        sources=a[3],original_serve_argv=a[0],only_mode_difference='TIANSHU_PAIR_LOAD_MODE',native_arguments_grid_forwarded_cases=n,
        future_reservation_mode='candidate',no_performance_or_quality_claim=True)
    (BASE/'PROFILE_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({key:value for key,value in gate.items() if key!='sources'}))
if __name__=='__main__':main()
