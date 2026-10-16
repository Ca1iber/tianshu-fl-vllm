# SPDX-License-Identifier: Apache-2.0
"""Screen static per-layer INT8 KV and folded scales; excludes writer costs."""
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
        scope='Manual static per-layer INT8 fixtures; mode1 is per-tensor dequant semantics on INT8 storage, NOT FP8 data. Folded candidate mode0 casts integer values and separately applies K/V scales. No real writer, model, service speed or Math claim.')
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
        for path in (Path(__file__),Path(native.__file__),Path(backend.__file__),Path(interface.__file__),Path(cache_module.__file__)):
            sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        data['source_snapshots']=sources
        data['profile_environment']=profile
        for name,sha in sources.items():
            target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
            assert hashlib.sha256(target.read_bytes()).hexdigest()==sha
        from fold_value_scale import load_reducer
        scaled_reduce,derivation=load_reducer(cache_module.__file__,root/'value_scale_generated')
        data['derivation']=derivation
        for extra in (Path(__file__).with_name('fold_value_scale.py'),Path(derivation['generated_source'])):
            extra=extra.resolve();sources[str(extra)]=hashlib.sha256(extra.read_bytes()).hexdigest()
            dest=root/'source_snapshot'/extra.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(extra.read_bytes())
        saved_kernel,saved_reduce=native.kernel_unified_attention,native.reduce_segments
        tile_state=dict(tile=16,folded=False,value_scale=None)
        class Producer:
            def __init__(self,tile,warps):self.tile=tile;self.warps=warps;self.metadata=None
            def __getitem__(self,grid):
                def launch(*a,**kw):
                    assert not a and kw['IS_3D'] and kw['HEAD_SIZE']==128 and kw['num_queries_per_kv']==8
                    assert kw['KV_QUANT_MODE'] in (0,1)
                    tile_state['tile']=self.tile
                    kw.update(TILE_SIZE=self.tile,num_warps=self.warps,num_stages=1,KV_CACHE_MODIFIER='.cv')
                    result=cache_module.kernel_unified_attention[grid](**kw)
                    self.metadata=dict(tile=self.tile,warps=self.warps,quant_mode=int(kw['KV_QUANT_MODE']),grid=list(grid),
                        n_regs=getattr(result,'n_regs',None),n_spills=getattr(result,'n_spills',None),
                        compiled_warps=result.metadata.num_warps,key_dtype=str(kw['key_cache_ptr'].dtype))
                    return result
                return launch
        class Reducer:
            def __getitem__(self,grid):
                def launch(*a,**kw):
                    kw['TILE_SIZE']=tile_state['tile']
                    if tile_state['folded']:
                        kw.update(v_tensor_scale_ptr=tile_state['value_scale'],APPLY_V_TENSOR_SCALE=True)
                        return scaled_reduce[grid](*a,**kw)
                    return cache_module.reduce_segments[grid](*a,**kw)
                return launch
        native.reduce_segments=Reducer()
        def check(a,b):
            diff=(a.float()-b.float()).abs();bad=diff>.02+.01*b.float().abs()
            return dict(passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),
                max_abs=diff.max().item(),failed_elements=bad.sum().item(),atol=.02,rtol=.01)
        def approximation(a,b):
            a,b=a.float(),b.float();diff=a-b
            return dict(max_abs=diff.abs().max().item(),relative_RMS=(diff.square().mean().sqrt()/b.square().mean().sqrt()).item())
        def quantize(x,scale):
            value=x.float()/scale
            rounded=torch.trunc(torch.where(value>=0,value+.5,value-.5))
            clips=int(((rounded<-128)|(rounded>127)).sum().item())
            return rounded.clamp(-128,127).to(torch.int8),clips
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
                k_scale_value=1/16;v_scale_value=1/16
                ks=torch.tensor([k_scale_value],device='cuda',dtype=torch.float32);vs=torch.tensor([v_scale_value],device='cuda',dtype=torch.float32)
                fixtures=[]
                plans=[('BF16',original_k,original_v,False,16,1),
                    ('INT8_native_tensor',packed_k,packed_v,False,16,1),
                    ('INT8_folded_w1',packed_k,packed_v,True,16,1),
                    ('INT8_folded_w2',packed_k,packed_v,True,32,2)]
                for name,k,v,folded,tile,warps in plans:
                    f=dict(name=name,k=k,v=v,folded=folded,producer=Producer(tile,warps),
                        mode=KVQuantMode.FP8_PER_TENSOR if name=='INT8_native_tensor' else KVQuantMode.NONE)
                    f['output']=torch.full((n+3,16,128),17.,device='cuda',dtype=q.dtype)
                    f['partial']=torch.full((n,16,64,128),17.,device='cuda',dtype=torch.float32)
                    f['max']=torch.full((n,16,64),17.,device='cuda',dtype=torch.float32);f['sum']=torch.full_like(f['max'],17.)
                    fixtures.append(f)
                quantization=[]
                def update_quant():
                    ki,kclips=quantize(original_k,k_scale_value);vi,vclips=quantize(original_v,v_scale_value)
                    packed_k.copy_(ki);packed_v.copy_(vi)
                    quantization.append(dict(K_clipped=kclips,V_clipped=vclips,elements_per_K_or_V=ki.numel(),K_scale=k_scale_value,V_scale=v_scale_value))
                def invoke(f):
                    native.unified_attention(q=q,k=f['k'],v=f['v'],out=f['output'],cu_seqlens_q=cu,
                        max_seqlen_q=1,seqused_k=used,max_seqlen_k=max(lengths),softmax_scale=128**-.5*(k_scale_value if f['folded'] else 1.),
                        causal=True,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=None,k_descale=ks if f['name']=='INT8_native_tensor' else None,v_descale=vs if f['name']=='INT8_native_tensor' else None,
                        seq_threshold_3D=64,num_par_softmax_segments=segments,softmax_segm_output=f['partial'],
                        softmax_segm_max=f['max'],softmax_segm_expsum=f['sum'],kv_quant_mode=f['mode'])
                update_quant();phases=[]
                for phase in range(2):
                    if phase:q.neg_();original_k.mul_(.5);original_v.neg_();update_quant()
                    expected,cpu=reference(q,original_k,original_v,table,lengths)
                    dequant_k=packed_k.float()*k_scale_value;dequant_v=packed_v.float()*v_scale_value
                    quant_expected,quant_cpu=reference(q,dequant_k,dequant_v,table,lengths)
                    checks=[]
                    for f in fixtures:
                        native.kernel_unified_attention=f['producer'];tile_state.update(folded=f['folded'],value_scale=vs)
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
                        result=check(f['output'][:n],expected if f['name']=='BF16' else quant_expected);assert result['passed'],(f['name'],result)
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
                    metadata=f['producer'].metadata,key_strides=list(f['k'].stride()),scale_folded=f['folded']) for i,f in enumerate(fixtures)]
                data['cases'].append(dict(batch=n,context=context,lengths=lengths,phases=phases,results=results,quantization=quantization));save()
                del fixtures;torch.cuda.empty_cache()
        native.kernel_unified_attention=saved_kernel;native.reduce_segments=saved_reduce
        for name,sha in sources.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
        data.update(phase='complete',complete=True,manual_quantization_not_native_writer=True,full_model_and_Math_not_run=True)
    except BaseException as error:data.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
    finally:signal.alarm(0);save()
    return 0 if data['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
