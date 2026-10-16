# SPDX-License-Identifier: Apache-2.0
"""Screen a software8-bit float KV codec using BF16 bit reconstruction."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
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
        scope='Manual page-interleaved software finite E4M3/scale1over16 uint8 KV, bit-reconstructed BF16 reads. Current attention arithmetic and reducer unchanged. No writer/allocator/model/Math/service or native FP8 hardware claim.')
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
        from triton.compiler.errors import CompilationError
        from triton.runtime.errors import OutOfResources
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
        from codec_variant import load_codec,decode_scalar,check_cpu,legacy,LEGACY_PATH
        gate=check_cpu();assert gate['complete'];data['CPU_gate']=gate
        soft_module,derivation=load_codec(cache_module.__file__,root/'soft8_generated');data['derivation']=derivation
        legacy_module,legacy_derivation=legacy.load_codec(cache_module.__file__,root/'legacy_soft8_generated');data['legacy_derivation']=legacy_derivation
        for extra in (Path(__file__).with_name('codec_variant.py'),LEGACY_PATH,LEGACY_PATH.with_name('CPU_GATE_32g.json'),Path(legacy_derivation['generated_source']),Path(__file__).with_name('CPU_GATE_32g.json'),Path(derivation['generated_source'])):
            extra=extra.resolve();sources[str(extra)]=hashlib.sha256(extra.read_bytes()).hexdigest()
            dest=root/'source_snapshot'/extra.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(extra.read_bytes())
        codes=torch.arange(256,device='cuda',dtype=torch.int32).to(torch.uint8);decoded=torch.empty(256,device='cuda',dtype=torch.bfloat16)
        soft_module._codec_all_codes[(1,)](codes,decoded,num_warps=1,num_stages=1);torch.cuda.synchronize()
        actual=decoded.cpu().tolist();expected=[decode_scalar(c) for c in range(256)]
        assert actual==expected and all(math.copysign(1.,a)==math.copysign(1.,b) for a,b in zip(actual,expected))
        data['all256_actual_GPU_codes']=actual;data['all256_GPU_decode_equal_CPU_reference']=True
        saved_kernel,saved_reduce=native.kernel_unified_attention,native.reduce_segments
        tile_state=dict(tile=16)
        class Producer:
            def __init__(self,tile,warps,name):self.tile=tile;self.warps=warps;self.name=name;self.metadata=None
            def __getitem__(self,grid):
                def launch(*a,**kw):
                    assert not a and kw['IS_3D'] and kw['HEAD_SIZE']==128 and kw['num_queries_per_kv']==8
                    assert kw['KV_QUANT_MODE']==0
                    tile_state['tile']=self.tile
                    kw.update(TILE_SIZE=self.tile,num_warps=self.warps,num_stages=1,KV_CACHE_MODIFIER='.cv')
                    kernel=(legacy_module if '_u16_' in self.name else soft_module).kernel_unified_attention if kw['key_cache_ptr'].dtype==torch.uint8 else cache_module.kernel_unified_attention
                    result=kernel[grid](**kw)
                    self.metadata=dict(codec_intermediates_bits=16 if '_u16_' in self.name else 32 if kw['key_cache_ptr'].dtype==torch.uint8 else None,tile=self.tile,warps=self.warps,quant_mode=int(kw['KV_QUANT_MODE']),grid=list(grid),
                        n_regs=getattr(result,'n_regs',None),n_spills=getattr(result,'n_spills',None),
                        compiled_warps=result.metadata.num_warps,key_dtype=str(kw['key_cache_ptr'].dtype))
                    return result
                return launch
        class Reducer:
            def __getitem__(self,grid):
                def launch(*a,**kw):
                    kw['TILE_SIZE']=tile_state['tile']
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
            assert scale==1/16
            bits=x.view(torch.int16).to(torch.int32);magnitude=bits&32767
            rounded=((magnitude+7+((magnitude>>4)&1))>>4)-928
            clips=int((rounded>127).sum().item())
            codes=torch.where(rounded>=8,rounded.clamp(max=127),0)|((bits&32768)>>8)
            return codes.to(torch.uint8),clips
        def dequantize(codes):
            codes=codes.to(torch.int32);magnitude=codes&127
            bits=torch.where(magnitude>=8,(magnitude<<4)+14848,0)|((codes&128)<<8)
            return bits.to(torch.int16).view(torch.bfloat16)
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
            for case_id,(n,context) in enumerate(((64,4096),(64,5120))):
                torch.manual_seed(2026101600+case_id)
                lengths=[max(1,context-i*37%min(context,257)) for i in range(n)]
                width=(max(lengths)+15)//16;pages=n*width
                table=torch.randperm(pages,device='cuda').int().reshape(n,width)
                q=torch.randn(n,16,128,device='cuda',dtype=torch.bfloat16)
                original_storage=torch.randn(pages,2,16,2,128,device='cuda',dtype=q.dtype)
                original_k,original_v=original_storage.unbind(1)
                cu=torch.arange(n+1,device='cuda',dtype=torch.int32);used=torch.tensor(lengths,device='cuda',dtype=torch.int32)
                segments=8 if n==64 else 16
                packed_storage=torch.empty((pages,2,16,2,128),device='cuda',dtype=torch.uint8)
                packed_k,packed_v=packed_storage.unbind(1)
                k_scale_value=1/16;v_scale_value=1/16
                fixtures=[]
                plans=[('BF16',original_k,original_v,16,1,8),
                       ('Soft8_u16_t16_w1_s8',packed_k,packed_v,16,1,8),
                       ('Soft8_u32_t16_w1_s8',packed_k,packed_v,16,1,8),
                       ('Soft8_u32_t16_w2_s8',packed_k,packed_v,16,2,8),
                       ('Soft8_u32_t32_w2_s8',packed_k,packed_v,32,2,8)]
                for name,k,v,tile,warps,splits in plans:
                    f=dict(name=name,k=k,v=v,segments=splits,producer=Producer(tile,warps,name),
                        mode=KVQuantMode.NONE)
                    f['output']=torch.full((n+3,16,128),17.,device='cuda',dtype=q.dtype)
                    f['partial']=torch.full((n,16,64,128),17.,device='cuda',dtype=torch.float32)
                    f['max']=torch.full((n,16,64),17.,device='cuda',dtype=torch.float32);f['sum']=torch.full_like(f['max'],17.)
                    fixtures.append(f)
                quantization=[]
                def update_quant():
                    ki,kclips=quantize(original_k,k_scale_value);vi,vclips=quantize(original_v,v_scale_value)
                    packed_k.copy_(ki);packed_v.copy_(vi)
                    quantization.append(dict(K_flushed=int(((ki&127)<8).sum().item()),V_flushed=int(((vi&127)<8).sum().item()),K_clipped=kclips,V_clipped=vclips,elements_per_K_or_V=ki.numel(),K_scale=k_scale_value,V_scale=v_scale_value))
                def invoke(f):
                    native.unified_attention(q=q,k=f['k'],v=f['v'],out=f['output'],cu_seqlens_q=cu,
                        max_seqlen_q=1,seqused_k=used,max_seqlen_k=max(lengths),softmax_scale=128**-.5,
                        causal=True,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=None,k_descale=None,v_descale=None,
                        seq_threshold_3D=64,num_par_softmax_segments=f['segments'],softmax_segm_output=f['partial'],
                        softmax_segm_max=f['max'],softmax_segm_expsum=f['sum'],kv_quant_mode=f['mode'])
                update_quant();phases=[]
                compiled=[];compile_failures=[]
                for f in fixtures:
                    native.kernel_unified_attention=f['producer']
                    try:
                        invoke(f);torch.cuda.synchronize()
                    except (CompilationError,OutOfResources) as error:
                        assert f['name']!='BF16', 'The mandatory current BF16 control must compile'
                        compile_failures.append(dict(name=f['name'],error=repr(error),traceback=traceback.format_exc()))
                    else:compiled.append(f)
                fixtures=compiled;assert fixtures and fixtures[0]['name']=='BF16'

                for phase in range(2):
                    if phase:q.neg_();original_k.mul_(.5);original_v.neg_();update_quant()
                    expected,cpu=reference(q,original_k,original_v,table,lengths)
                    dequant_k=dequantize(packed_k);dequant_v=dequantize(packed_v)
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
                            assert bool((f[field].flatten()[n*16*f['segments']*scale:]==17.).all())
                        result=check(f['output'][:n],expected if f['name']=='BF16' else quant_expected);assert result['passed'],(f['name'],result)
                        checks.append(dict(name=f['name'],against_own_FP32=result,against_BF16_FP32=approximation(f['output'][:n],expected),tails_preserved=True))
                    u16=next(f for f in fixtures if f['name']=='Soft8_u16_t16_w1_s8')
                    u32=next(f for f in fixtures if f['name']=='Soft8_u32_t16_w1_s8')
                    same=torch.equal(u16['output'],u32['output']);assert same
                    phases.append(dict(same_format_wordwidth_outputs_bitwise_equal=same,phase=phase,checks=checks,CPU_BF16_reference=cpu,CPU_dequantized_reference=quant_cpu,outputs_cleared_before_replay=True))
                timings=[[] for _ in fixtures]
                for repeat in range(7):
                    order=list(range(len(fixtures)));offset=repeat%len(fixtures)
                    for i in order[offset:]+order[:offset]:
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                        for _ in range(20):fixtures[i]['graph'].replay()
                        b.record();b.synchronize();timings[i].append(a.elapsed_time(b)/20)
                medians=list(map(statistics.median,timings))
                results=[dict(name=f['name'],median_ms=medians[i],raw_ms=timings[i],time_change_percent=100*(medians[i]/medians[0]-1),
                    metadata=f['producer'].metadata,key_strides=list(f['k'].stride()),scale_folded=False,segments=f['segments']) for i,f in enumerate(fixtures)]
                data['cases'].append(dict(batch=n,context=context,lengths=lengths,phases=phases,results=results,quantization=quantization,compile_failures=compile_failures));save()
                del fixtures;torch.cuda.empty_cache()
        native.kernel_unified_attention=saved_kernel;native.reduce_segments=saved_reduce
        for name,sha in sources.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
        data.update(phase='complete',complete=True,manual_quantization_not_native_writer=True,full_model_and_Math_not_run=True,software_custom_finite_E4M3=True,not_native_FP8_hardware=True)
    except BaseException as error:data.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
    finally:signal.alarm(0);save()
    return 0 if data['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
