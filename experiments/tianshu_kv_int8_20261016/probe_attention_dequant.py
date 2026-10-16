# SPDX-License-Identifier: Apache-2.0
"""Compare INT8 S/P scaling to BF16 dequantization on load; attention only."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True)
    args=parser.parse_args();root=args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>60
    signal.alarm(min(1800,remaining))
    data=dict(phase='preparing',complete=False,runner_pid=os.getpid(),official_benchmark=False,adopted=False,cases=[],source_snapshots={},
        scope='Manual quantized fixtures; attention plus reducer only. No real KV writer, service speed or Math claim.')
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
        os.environ.clear();os.environ.update(environment)
        runpy.run_path(str(Path(profile['PYTHONPATH'])/'sitecustomize.py'))
        import torch
        import vllm_fl
        import vllm.v1.attention.ops.triton_unified_attention as native
        import vllm.v1.attention.backends.triton_attn as backend
        import vllm.v1.kv_cache_interface as interface
        from vllm.v1.kv_cache_interface import KVQuantMode
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        cache_module=sys.modules['tianshu_native_cache_policy']
        assert hashlib.sha256(Path(cache_module.__file__).read_bytes()).hexdigest()=='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'
        from dequant_on_load import load_kernel
        changed_kernel,derivation=load_kernel(Path(cache_module.__file__),root/'dequant_generated')
        data['derivation']=derivation
        for path in (Path(__file__).with_name('dequant_on_load.py'),Path(derivation['generated_source']),Path(__file__),Path(native.__file__),Path(backend.__file__),Path(interface.__file__),Path(cache_module.__file__)):
            sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        data['source_snapshots']=sources
        data['profile_environment']=profile
        for name,sha in sources.items():
            target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
            assert hashlib.sha256(target.read_bytes()).hexdigest()==sha
        saved_kernel,saved_reduce=native.kernel_unified_attention,native.reduce_segments
        tile_state=dict(tile=16)
        class Producer:
            def __init__(self,tile,warps,onload):self.tile=tile;self.warps=warps;self.onload=onload;self.metadata=None
            def __getitem__(self,grid):
                def launch(*a,**kw):
                    assert not a and kw['IS_3D'] and kw['HEAD_SIZE']==128 and kw['num_queries_per_kv']==8
                    assert kw['KV_QUANT_MODE'] in (0,2)
                    tile_state['tile']=self.tile
                    kw.update(TILE_SIZE=self.tile,num_warps=self.warps,num_stages=1,KV_CACHE_MODIFIER='.cv')
                    kernel=changed_kernel if self.onload else cache_module.kernel_unified_attention
                    if self.onload:kw['DEQUANT_KV_ON_LOAD']=True
                    result=kernel[grid](**kw)
                    self.metadata=dict(onload_dequant=self.onload,tile=self.tile,warps=self.warps,quant_mode=int(kw['KV_QUANT_MODE']),grid=list(grid),
                        n_regs=getattr(result,'n_regs',None),n_spills=getattr(result,'n_spills',None),
                        compiled_warps=result.metadata.num_warps,key_dtype=str(kw['key_cache_ptr'].dtype))
                    return result
                return launch
        class Reducer:
            def __getitem__(self,grid):
                def launch(*a,**kw):
                    kw['TILE_SIZE']=tile_state['tile'];return cache_module.reduce_segments[grid](*a,**kw)
                return launch
        native.reduce_segments=Reducer()
        def check(a,b):
            diff=(a.float()-b.float()).abs();bad=diff>.02+.01*b.float().abs()
            return dict(passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),
                max_abs=diff.max().item(),failed_elements=bad.sum().item(),atol=.02,rtol=.01)
        def approximation(a,b):
            a,b=a.float(),b.float();diff=a-b
            return dict(max_abs=diff.abs().max().item(),relative_RMS=(diff.square().mean().sqrt()/b.square().mean().sqrt()).item())
        def quantize(x):
            x=x.float();scale=(x.abs().amax(-1)/127).clamp_min(1e-6)
            value=x/scale[...,None]
            value=torch.trunc(torch.where(value>=0,value+.5,value-.5)).clamp(-128,127).to(torch.int8)
            return value,scale
        def reference(q,k,v,table,lengths):
            results=[];cpu=[]
            for i,length in enumerate(lengths):
                ids=table[i,:(length+15)//16].long()
                keys=k[ids].reshape(-1,2,128)[:length].float().repeat_interleave(8,1)
                values=v[ids].reshape(-1,2,128)[:length].float().repeat_interleave(8,1)
                scores=torch.bmm(q[i].float()[:,None],keys.permute(1,2,0))[:,0]*128**-.5
                result=torch.bmm(scores.softmax(-1)[:,None],values.permute(1,0,2))[:,0];results.append(result)
                if i in (0,len(lengths)//2,len(lengths)-1):
                    kc=keys.cpu();vc=values.cpu();qc=q[i].cpu().float()
                    for head in (0,7,8,15):
                        exact=((kc[:,head]@qc[head])*128**-.5).softmax(0)@vc[:,head]
                        cpu.append(check(result[head].cpu(),exact))
            assert all(c['passed'] for c in cpu)
            return torch.stack(results),cpu
        data['phase']='numerical';save()
        with torch.inference_mode():
            for case_id,(n,context) in enumerate(((8,33),(64,4096),(64,5120))):
                torch.manual_seed(2026101600+case_id)
                lengths=[max(1,context-i*37%min(context,257)) for i in range(n)]
                width=(max(lengths)+15)//16;pages=n*width
                table=torch.randperm(pages,device='cuda').int().reshape(n,width)
                q=torch.randn(n,16,128,device='cuda',dtype=torch.bfloat16)
                original_k=torch.randn(pages,16,2,128,device='cuda',dtype=q.dtype);original_v=torch.randn_like(original_k)
                cu=torch.arange(n+1,device='cuda',dtype=torch.int32);used=torch.tensor(lengths,device='cuda',dtype=torch.int32)
                segments=8 if n==64 else 16
                packed_k=torch.empty_like(original_k,dtype=torch.int8);packed_v=torch.empty_like(packed_k)
                packed_ks=torch.empty((pages,16,2),device='cuda',dtype=torch.float32);packed_vs=torch.empty_like(packed_ks)
                inline=torch.full((pages,2,16,2,132),17,device='cuda',dtype=torch.int8)
                impl=object.__new__(backend.TritonAttentionImpl);impl._k_scale_cache=impl._v_scale_cache=None
                impl._ensure_scale_caches(inline)
                ik,iv=inline.unbind(1)
                fixtures=[]
                plans=[('BF16',original_k,original_v,None,None,16,1,False),
                    ('INT8_packed128_original',packed_k,packed_v,packed_ks,packed_vs,16,1,False),
                    ('INT8_packed128_load_w1',packed_k,packed_v,packed_ks,packed_vs,16,1,True),
                    ('INT8_packed128_load_w2',packed_k,packed_v,packed_ks,packed_vs,32,2,True),
                    ('INT8_inline132_load_w1',ik,iv,impl._k_scale_cache,impl._v_scale_cache,16,1,True)]
                for name,k,v,ks,vs,tile,warps,onload in plans:
                    f=dict(name=name,k=k,v=v,ks=ks,vs=vs,producer=Producer(tile,warps,onload),
                        mode=KVQuantMode.NONE if ks is None else KVQuantMode.INT8_PER_TOKEN_HEAD)
                    f['output']=torch.full((n+3,16,128),17.,device='cuda',dtype=q.dtype)
                    f['partial']=torch.full((n,16,64,128),17.,device='cuda',dtype=torch.float32)
                    f['max']=torch.full((n,16,64),17.,device='cuda',dtype=torch.float32);f['sum']=torch.full_like(f['max'],17.)
                    fixtures.append(f)
                def update_quant():
                    ki,ks=quantize(original_k);vi,vs=quantize(original_v)
                    packed_k.copy_(ki);packed_v.copy_(vi);packed_ks.copy_(ks);packed_vs.copy_(vs)
                    ik[...,:128].copy_(ki);iv[...,:128].copy_(vi);impl._k_scale_cache.copy_(ks);impl._v_scale_cache.copy_(vs)
                def invoke(f):
                    native.unified_attention(q=q,k=f['k'],v=f['v'],out=f['output'],cu_seqlens_q=cu,
                        max_seqlen_q=1,seqused_k=used,max_seqlen_k=max(lengths),softmax_scale=128**-.5,
                        causal=True,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=None,k_descale=None,v_descale=None,
                        seq_threshold_3D=64,num_par_softmax_segments=segments,softmax_segm_output=f['partial'],
                        softmax_segm_max=f['max'],softmax_segm_expsum=f['sum'],kv_quant_mode=f['mode'],k_scale_cache=f['ks'],v_scale_cache=f['vs'])
                update_quant();phases=[]
                for phase in range(2):
                    if phase:q.neg_();original_k.mul_(.5);original_v.neg_();update_quant()
                    expected,cpu=reference(q,original_k,original_v,table,lengths)
                    dequant_k=packed_k.float()*packed_ks[...,None];dequant_v=packed_v.float()*packed_vs[...,None]
                    quant_expected,quant_cpu=reference(q,dequant_k,dequant_v,table,lengths)
                    checks=[]
                    for f in fixtures:
                        native.kernel_unified_attention=f['producer']
                        if not phase:
                            for _ in range(3):invoke(f)
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):invoke(f)
                            f['graph']=graph
                        f['output'].fill_(17.)
                        f['graph'].replay();torch.cuda.synchronize()
                        assert bool((f['output'][n:]==17.).all())
                        for field,scale in (('partial',128),('max',1),('sum',1)):
                            assert bool((f[field].flatten()[n*16*segments*scale:]==17.).all())
                        result=check(f['output'][:n],expected if f['ks'] is None else quant_expected);assert result['passed'],(f['name'],result)
                        checks.append(dict(name=f['name'],against_own_FP32=result,against_BF16_FP32=approximation(f['output'][:n],expected),tails_preserved=True))
                    phases.append(dict(phase=phase,checks=checks,CPU_BF16_reference=cpu,CPU_dequantized_reference=quant_cpu,outputs_cleared_before_replay=True))
                timings=[[] for _ in fixtures]
                for repeat in range(7):
                    order=list(range(len(fixtures)));offset=repeat%len(fixtures)
                    for i in order[offset:]+order[:offset]:
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                        for _ in range(20):fixtures[i]['graph'].replay()
                        b.record();b.synchronize();timings[i].append(a.elapsed_time(b)/20)
                medians=list(map(statistics.median,timings))
                results=[dict(name=f['name'],median_ms=medians[i],raw_ms=timings[i],time_change_percent=100*(medians[i]/medians[0]-1),
                    metadata=f['producer'].metadata,key_strides=list(f['k'].stride()),scale_strides=None if f['ks'] is None else list(f['ks'].stride())) for i,f in enumerate(fixtures)]
                data['cases'].append(dict(batch=n,context=context,lengths=lengths,phases=phases,results=results));save()
                del fixtures;torch.cuda.empty_cache()
        native.kernel_unified_attention=saved_kernel;native.reduce_segments=saved_reduce
        for name,sha in sources.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
        data.update(phase='complete',complete=True,manual_quantization_not_native_writer=True,full_model_and_Math_not_run=True)
    except BaseException as error:data.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
    finally:signal.alarm(0);save()
    return 0 if data['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
