# SPDX-License-Identifier: Apache-2.0
"""Current best CV/KV16/S8 attention, direct sequence mapping component check."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import statistics
import subprocess
import sys
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True)
    args=p.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>60
    signal.alarm(min(1800,remaining))
    data=dict(phase='preparing',complete=False,runner_pid=os.getpid(),official_benchmark=False,adopted=False,cases=[],source_snapshots={},
        scope='Port preserved direct mapping to current CV BF16 producer. Same attention/reducer/segments, no quantization/model/Math/service.')
    def save():
        p=root/'result.tmp';p.write_text(json.dumps(data,indent=2)+'\n');p.replace(root/'result.json')
        print(json.dumps(dict(phase=data['phase'],cases=len(data['cases']),error=data.get('error'))),flush=True)
    save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
        (root/'hardware_before.txt').write_text(hardware)
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tianshu_gemm_topp_geo_20261016'))
        from validation_profile_new import prepare
        command,environment,profile,sources=prepare(root,Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
        os.environ.clear();os.environ.update(environment);runpy.run_path(str(Path(profile['PYTHONPATH'])/'sitecustomize.py'))
        import torch
        import vllm_fl
        import vllm.v1.attention.ops.triton_unified_attention as native
        import vllm.v1.attention.ops.triton_attention_helpers as helpers
        import vllm.v1.attention.backends.triton_attn as backend
        from render import load_kernel
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        cache=sys.modules['tianshu_native_cache_policy'];candidate,derivation=load_kernel(cache.__file__,root/'direct_generated')
        data['derivation']=derivation;data['profile_environment']=profile
        for path in (Path(__file__),Path(__file__).with_name('render.py'),Path(native.__file__),Path(helpers.__file__),Path(backend.__file__),Path(cache.__file__),Path(derivation['generated_source'])):
            sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        data['source_snapshots']=sources
        for name,sha in sources.items():
            dest=root/'source_snapshot'/Path(name).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
            assert hashlib.sha256(dest.read_bytes()).hexdigest()==sha
        original_producer,original_reducer=native.kernel_unified_attention,native.reduce_segments
        state=dict(tile=16)
        class Producer:
            def __init__(self,direct,tile,warps):self.direct=direct;self.tile=tile;self.warps=warps;self.metadata=None
            def __getitem__(self,grid):
                def invoke(*args,**kw):
                    assert not args and kw['IS_3D'] and kw['HEAD_SIZE']==128 and kw['BLOCK_Q']==2 and kw['num_queries_per_kv']==8
                    state['tile']=self.tile;kw.update(TILE_SIZE=self.tile,num_warps=self.warps,num_stages=1,KV_CACHE_MODIFIER='.cv')
                    actual=(kw['num_seqs'],grid[1],grid[2]) if self.direct else grid
                    if self.direct:kw['DIRECT_DECODE']=True
                    compiled=(candidate if self.direct else cache.kernel_unified_attention)[actual](**kw)
                    self.metadata=dict(original_grid=list(grid),actual_grid=list(actual),tile=self.tile,warps=self.warps,
                        n_regs=compiled.n_regs,n_spills=compiled.n_spills,compiled_warps=compiled.metadata.num_warps)
                    return compiled
                return invoke
        class Reducer:
            def __getitem__(self,grid):
                def invoke(*args,**kw):
                    kw['TILE_SIZE']=state['tile'];return cache.reduce_segments[grid](*args,**kw)
                return invoke
        native.reduce_segments=Reducer()
        def check(a,b):
            a,b=a.float(),b.float();delta=(a-b).abs();bad=delta>.02+.01*b.abs()
            return dict(passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),max_abs=delta.max().item(),failed_elements=bad.sum().item(),atol=.02,rtol=.01)
        scenarios=((1,33,False,False),(8,4096,False,False),(24,4096,True,False),(32,4096,False,False),
                   (64,4096,False,False),(64,5120,True,False),(8,257,True,True),(64,4096,True,True))
        data['phase']='numerical';save()
        with torch.inference_mode():
            for case_id,(n,context,strided,holes) in enumerate(scenarios):
                torch.manual_seed(2026101600+case_id)
                qlens=[int(i%3!=1) if holes else 1 for i in range(n)];queries=sum(qlens)
                lengths=[max(1,context-i*37%min(context,257)) for i in range(n)]
                width=(max(lengths)+15)//16;pages=n*width
                table=torch.randperm(pages,device='cuda').int().reshape(n,width)
                q=torch.randn(queries,20 if strided else 16,128,device='cuda',dtype=torch.bfloat16)[:,:16]
                storage=torch.randn(pages*(2 if strided else 1),2,16,2,128,device='cuda',dtype=q.dtype)
                if strided:storage=storage[::2]
                k,v=storage.unbind(1)
                def make_cu(lens):return torch.tensor([0]+lens,device='cuda',dtype=torch.int32).cumsum(0).int()
                cu=make_cu(qlens);used=torch.tensor(lengths,device='cuda',dtype=torch.int32)
                tile=16 if n in (8,16,32,40,48,56,64) else 32
                warps=1 if n in (8,16,32,40,48,56,64) else 2 if 8<=n<=64 else 4
                segments=64 if n==1 else 8 if n==64 else 4 if 16<=n<=64 else 16
                fixtures=[]
                for direct in (False,True):
                    fixtures.append(dict(direct=direct,producer=Producer(direct,tile,warps),out=torch.full((queries+3,16,128),17.,device='cuda',dtype=q.dtype),
                        partial=torch.full((queries,16,64,128),17.,device='cuda',dtype=torch.float32),
                        maximum=torch.full((queries,16,64),17.,device='cuda',dtype=torch.float32),expsum=torch.full((queries,16,64),17.,device='cuda',dtype=torch.float32)))
                def invoke(f):
                    native.unified_attention(q=q,k=k,v=v,out=f['out'],cu_seqlens_q=cu,max_seqlen_q=1,seqused_k=used,
                        max_seqlen_k=max(lengths),softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=table,softcap=0.,
                        q_descale=None,k_descale=None,v_descale=None,seq_threshold_3D=64,num_par_softmax_segments=segments,
                        softmax_segm_output=f['partial'],softmax_segm_max=f['maximum'],softmax_segm_expsum=f['expsum'])
                phases=[]
                for phase in range(2):
                    if phase:
                        q.neg_();k.mul_(.5);v.neg_()
                        if holes:qlens=qlens[-1:]+qlens[:-1];cu.copy_(make_cu(qlens))
                    expected=[];cpu=[];query_index=0
                    for seq,qlen in enumerate(qlens):
                        if not qlen:continue
                        ids=table[seq,:(lengths[seq]+15)//16].long()
                        keys=k[ids].reshape(-1,2,128)[:lengths[seq]].float().repeat_interleave(8,1)
                        values=v[ids].reshape(-1,2,128)[:lengths[seq]].float().repeat_interleave(8,1)
                        scores=torch.bmm(q[query_index].float()[:,None],keys.permute(1,2,0))[:,0]*128**-.5
                        answer=torch.bmm(scores.softmax(-1)[:,None],values.permute(1,0,2))[:,0];expected.append(answer)
                        if query_index in (0,queries//2,queries-1):
                            kc,vc,qc=keys.cpu(),values.cpu(),q[query_index].cpu().float()
                            for head in (0,7,8,15):cpu.append(check(answer[head].cpu(),((kc[:,head]@qc[head])*128**-.5).softmax(0)@vc[:,head]))
                        query_index+=1
                    expected=torch.stack(expected);assert all(x['passed'] for x in cpu)
                    checks=[]
                    for f in fixtures:
                        native.kernel_unified_attention=f['producer']
                        if not phase:
                            for _ in range(3):invoke(f)
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):invoke(f)
                            f['graph']=graph
                        f['out'].fill_(17.);f['graph'].replay();torch.cuda.synchronize()
                        assert bool((f['out'][queries:]==17.).all())
                        for field,mul in (('partial',128),('maximum',1),('expsum',1)):
                            assert bool((f[field].flatten()[queries*16*segments*mul:]==17.).all())
                        c=check(f['out'][:queries],expected);assert c['passed'],c
                        checks.append(dict(direct=f['direct'],against_FP32=c,tails_preserved=True))
                    bitwise=torch.equal(fixtures[0]['out'],fixtures[1]['out']);assert bitwise
                    phases.append(dict(phase=phase,query_lengths=list(qlens),CPU_reference=cpu,checks=checks,bitwise_equal=bitwise,outputs_cleared_before_replay=True))
                timings=[[],[]]
                for repeat in range(7):
                    for i in ((0,1) if repeat%2==0 else (1,0)):
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                        for _ in range(20):fixtures[i]['graph'].replay()
                        b.record();b.synchronize();timings[i].append(a.elapsed_time(b)/20)
                medians=list(map(statistics.median,timings))
                data['cases'].append(dict(batch=n,context=context,strided=strided,holes=holes,segments=segments,queries=queries,
                    key_strides=list(k.stride()),query_strides=list(q.stride()),phases=phases,raw_ms=timings,baseline_ms=medians[0],candidate_ms=medians[1],
                    time_change_percent=100*(medians[1]/medians[0]-1),producer_metadata=[f['producer'].metadata for f in fixtures]));save()
                del fixtures;torch.cuda.empty_cache()
        native.kernel_unified_attention=original_producer;native.reduce_segments=original_reducer
        for name,sha in sources.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
        data.update(phase='complete',complete=True,full_model_and_Math_not_run=True)
    except BaseException as error:data.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
    finally:signal.alarm(0);save()
    return 0 if data['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
