# SPDX-License-Identifier: Apache-2.0
"""Frozen PV primitive in full native attention; writer included in every timing."""
import argparse,gc,hashlib,inspect,json,logging,os,runpy,signal,statistics,sys
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import source
from layout import MODES,allocate
BASE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200
    signal.alarm(min(remaining,1800))
    state=dict(complete=False,phase='preparing',adopted=False,official_benchmark=False,cases=[],producer_records={},
        full_engine_allocator_not_installed=True,prefill_not_tested=True,Math_not_run=True,
        scope='Real native page layout, writer+paired-K decode; not end-to-end throughput')
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'result.json')
    save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
        sys.path.insert(0,str(BASE.parent/'tianshu_gemm_topp_geo_20261016'))
        from validation_profile_new import prepare
        command,environment,profile,sources=prepare(root,Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
        os.environ.clear();os.environ.update(environment);runpy.run_path(str(Path(profile['PYTHONPATH'])/'sitecustomize.py'))
        import torch
        import vllm_fl
        from vllm.v1.kv_cache_interface import FullAttentionSpec,KVQuantMode
        from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
        from vllm_fl.worker import iluvatar_adaptive_decode as adaptive
        from vllm_fl.worker import iluvatar_hybrid_attention as hybrid
        from types import SimpleNamespace
        config=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=2048,num_hidden_layers=42,
            num_attention_heads=16,num_key_value_heads=2),dtype=torch.bfloat16,quantization=None,is_multimodal_model=False),
            parallel_config=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,decode_context_parallel_size=1),lora_config=None,speculative_config=None)
        assert adaptive.configure(config,logging.getLogger('k8vi8_heads')) and hybrid.configure(config,logging.getLogger('k8vi8_heads'))
        sys.path.insert(0,str(BASE));import writer,reference_quantize
        reader,helper,proof=source.load_modules(root/'reader_generated')
        gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete']
        for path,digest in gate['sources'].items():assert sha(path)==digest;sources[path]=digest
        paths=[BASE/'CPU_GATE_32g.json',Path(reader.__file__),Path(inspect.getsourcefile(FullAttentionSpec)),Path(inspect.getsourcefile(_reshape_attention_kv_cache))]
        sources.update({str(path):sha(path) for path in paths})
        for path,digest in sources.items():
            dest=root/'source_snapshot'/Path(path).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(path).read_bytes());assert sha(dest)==digest
        state.update(source_snapshots=sources,profile_environment=profile,server_command=command,codec_proof=proof,modes=list(MODES),phase='probing');save()
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        def encode(x):
            bits=x.view(torch.int16).to(torch.int32);mag=bits&32767;rounded=(mag+7+((mag>>4)&1))>>4
            return (torch.where(rounded>=936,(rounded-928).clamp(max=127),0)|((bits&32768)>>8)).to(torch.uint8)
        def decode(x):
            bits=x.to(torch.int32);mag=bits&127
            return (torch.where(mag>=8,(mag<<4)+14848,0)|((bits&128)<<8)).to(torch.int16).view(torch.bfloat16)
        def quant(x):
            return reference_quantize.quantize(x)
        def check(actual,reference):
            diff=(actual.float()-reference).abs();bad=diff>.02+.01*reference.abs()
            return dict(passed=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all() and not bad.any()),failed_elements=int(bad.sum()),max_abs=float(diff.max()),atol=.02,rtol=.01)
        current=[0]
        class Producer:
            def __init__(self,kernel):self.kernel=kernel;self.mode=None;self.tile=None
            def __getitem__(self,grid):
                def launch(*args,**kw):
                    assert not args and kw['KV_QUANT_MODE']==(2 if self.mode!='K8V16' else 0)
                    n=kw['num_seqs'];launch_grid=grid
                    if kw['IS_3D']:kw.update(TILE_SIZE=int(self.mode[2:]) if self.mode.startswith('PV') else 16,num_warps=2 if self.mode.startswith('PV') else 1,num_stages=1)
                    else:
                        kw.update(BLOCK_M=16,BLOCK_Q=2,TILE_SIZE=16,num_warps=1,num_stages=1);launch_grid=(kw['query_ptr'].shape[0]//2+n,grid[1])
                    kw['PV_INT8']=self.mode.startswith('PV')
                    kw['KV_CACHE_MODIFIER']='.cv';self.tile=kw['TILE_SIZE'];compiled=self.kernel[launch_grid](**kw)
                    state['producer_records'][str((self.mode,n,current[0]))]=dict(mode=self.mode,batch=n,context=current[0],
                        key_dtype=str(kw['key_cache_ptr'].dtype),value_dtype=str(kw['value_cache_ptr'].dtype),KV_QUANT_MODE=int(kw['KV_QUANT_MODE']),
                        IS_3D=kw['IS_3D'],PV_INT8=kw['PV_INT8'],segments=kw['NUM_SEGMENTS_PER_SEQ'],tile=self.tile,warps=compiled.metadata.num_warps,
                        n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),binary_sha256=hashlib.sha256(compiled.kernel).hexdigest())
                    return compiled
                return launch
        producer=Producer(reader.kernel_unified_attention);raw_reduce=reader.reduce_segments
        class Reducer:
            def __getitem__(self,grid):
                def launch(*args,**kw):kw['TILE_SIZE']=producer.tile;return raw_reduce[grid](*args,**kw)
                return launch
        reader.kernel_unified_attention=producer;reader.reduce_segments=Reducer()
        cases=((1,33),(8,257),(40,4096),(64,4096),(64,5120))
        with torch.inference_mode():
          for case_id,(n,context) in enumerate(cases):
            torch.manual_seed(202610210100+case_id);current[0]=context
            lengths=[max(1,context-i*37%min(context,257)) for i in range(n)];width=(context+15)//16;pages=n*width
            logical_k=torch.randn(n,width*16,2,128,device='cuda',dtype=torch.bfloat16);logical_v=torch.randn_like(logical_k)
            logical_v[0,lengths[0]-1,0].zero_();logical_v[0,lengths[0]-1,1].zero_()
            logical_v[0,lengths[0]-1,1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],device='cuda',dtype=torch.bfloat16)
            qs=torch.randn(n+3,20,128,device='cuda',dtype=torch.bfloat16);q=qs[:n,:16]
            cu=torch.arange(n+1,device='cuda',dtype=torch.int32);used=torch.tensor(lengths,device='cuda',dtype=torch.int32)
            common_table=torch.randperm(pages,device='cuda').int().reshape(n,width)
            segments=(4 if context<=128 else 64 if n==1 else 32 if n<=3 else 16 if n<=8 else 8 if n<=16 else 16 if n<=24 else 4) if n<=64 else adaptive.choose_segments(n)
            if 33<=n<=56 and context>8192:segments=16
            fixtures=[]
            for mode in MODES:
                physical=(pages+2)*(2 if case_id%2 else 1)
                storage,native,k,v,ks,vs,layout=allocate(torch,FullAttentionSpec,_reshape_attention_kv_cache,mode,physical,bool(case_id%2))
                out=torch.full((n+3,20,128),17.,device='cuda',dtype=q.dtype);inputs=torch.empty_like(out)
                fixtures.append(dict(mode=mode,storage=storage,native=native,k=k,v=v,ks=ks,vs=vs,layout=layout,table=common_table.clone(),
                    output_storage=out,output=out[:,:16],input_storage=inputs,key=inputs[:,16:18],value=inputs[:,18:20],
                    slots=torch.full((n+3,),-1,device='cuda',dtype=torch.int64),partial=torch.full((n,16,64,128),17.,device='cuda',dtype=torch.float32),
                    maximum=torch.full((n,16,64),17.,device='cuda',dtype=torch.float32),total=torch.full((n,16,64),17.,device='cuda',dtype=torch.float32)))
            def refill(f):
                table=f['table'].flatten().long();f['k'][table]=encode(logical_k).reshape(pages,16,2,128)
                if f['mode']!='K8V16':
                    vq,vscale=quant(logical_v);f['v'][table]=vq.reshape(pages,16,2,128)
                    f['ks'][table]=1.;f['vs'][table]=vscale.reshape(pages,16,2)
                else:f['v'][table]=(encode(logical_v) if f['v'].dtype==torch.uint8 else logical_v).reshape(pages,16,2,128)
                f['expected_storage']=f['storage'].clone()
                for i,length in enumerate(lengths):
                    page,token=divmod(length-1,16);physical_page=f['table'][i,page]
                    f['key'][i].copy_(logical_k[i,length-1]);f['value'][i].copy_(logical_v[i,length-1]);f['slots'][i]=physical_page.long()*16+token
                    f['k'][physical_page,token]=255;f['v'][physical_page,token]=117
                    if f['ks'] is not None:f['ks'][physical_page,token]=111.;f['vs'][physical_page,token]=333.
                f['key'][n:].fill_(19.);f['value'][n:].fill_(23.)
                f['slots'][n:]=torch.tensor([-1,-33,-999],device='cuda',dtype=torch.int64)
            def invoke(f):
                producer.mode=f['mode'];writer.write(f['key'],f['value'],f['k'],f['v'],f['slots'],f['ks'],f['vs'])
                reader.unified_attention(q=q,k=f['k'],v=f['v'],out=f['output'],cu_seqlens_q=cu,max_seqlen_q=1,seqused_k=used,
                    max_seqlen_k=context,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=f['table'],softcap=0.,
                    q_descale=None,k_descale=None,v_descale=None,seq_threshold_3D=64,num_par_softmax_segments=segments,
                    softmax_segm_output=f['partial'],softmax_segm_max=f['maximum'],softmax_segm_expsum=f['total'],
                    kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD if f['mode']!='K8V16' else KVQuantMode.NONE,
                    k_scale_cache=f['ks'],v_scale_cache=f['vs'])
            phases=[]
            for phase in range(3):
                if phase:
                    q.neg_();logical_k.mul_(.5);logical_v.mul_(-.5);lengths=[max(1,length-7) for length in lengths]
                    used.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
                checks=[]
                for f in fixtures:
                    refill(f)
                    if phase==0:
                        for _ in range(3):invoke(f)
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):invoke(f)
                        f['graph']=graph
                    tensors=(q,f['key'],f['value'],f['slots'],used,f['table'])
                    before=[x.clone() for x in tensors];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                    f['graph'].replay();torch.cuda.synchronize()
                    assert all(torch.equal(x,y) for x,y in zip(tensors,before))
                    assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
                    assert torch.equal(f['storage'],f['expected_storage']),(case_id,phase,f['mode'],'writer physical bytes')
                    assert bool((f['output_storage'][n:]==17.).all() and (f['output_storage'][:n,16:]==17.).all())
                    count=n*16*segments if n<=64 else 0
                    for field,mult in [('partial',128),('maximum',1),('total',1)]:assert bool((f[field].flatten()[count*mult:]==17.).all())
                    expected=[];cpu_checks=[];precision_checks=[]
                    for i,length in enumerate(lengths):
                        keys=decode(encode(logical_k[i,:length])).float().repeat_interleave(8,1)
                        values=logical_v[i,:length]
                        if f['v'].dtype==torch.uint8:values=decode(encode(values)).float()
                        elif f['v'].dtype==torch.int8:
                            vq,vscale=quant(values);values=vq.float()*vscale[...,None]
                        else:values=values.float()
                        values=values.repeat_interleave(8,1)
                        scores=(keys.permute(1,0,2)@q[i].float().unsqueeze(-1)).squeeze(-1)*128**-.5
                        answer=(scores.softmax(-1).unsqueeze(1)@values.transpose(0,1)).squeeze(1);expected.append(answer)
                        if i in (0,n//2,n-1):
                            for head in (0,7,8,15):
                                kc,vc,qc=keys[:,head].cpu().double(),values[:,head].cpu().double(),q[i,head].cpu().double()
                                reference=((kc@qc)*128**-.5).softmax(0)@vc;actual=f['output'][i,head].cpu().float()
                                c=check(actual.double(),reference);c.update(CPU_FP64_reference=True,sequence=i,head=head,actual=actual.tolist(),expected=reference.tolist());assert c['passed'],(case_id,phase,f['mode'],c);cpu_checks.append(c)
                                original_v=logical_v[i,:length,head//8].cpu().double();original_k=logical_k[i,:length,head//8].cpu().double()
                                original_answer=((original_k@qc)*128**-.5).softmax(0)@original_v
                                precision_checks.append(dict(sequence=i,head=head,relative_L2=float(torch.linalg.vector_norm(actual-original_answer)/torch.linalg.vector_norm(original_answer).clamp_min(1.e-8)),max_abs=float((actual-original_answer).abs().max())))
                    c=check(f['output'][:n],torch.stack(expected));assert c['passed'],(case_id,phase,f['mode'],c)
                    writer_file=None
                    if f['mode']!='K8V16':
                        slots=f['slots'][:n];blocks=slots//16;tokens=slots%16
                        writer_file=f'writer_{f['mode']}_case{case_id}_phase{phase}.pt'
                        torch.save(dict(value_input=f['value'][:n].cpu(),actual_V_INT8=f['v'][blocks,tokens].cpu(),actual_V_scale=f['vs'][blocks,tokens].cpu(),actual_K_scale=f['ks'][blocks,tokens].cpu()),root/writer_file)
                    checks.append(dict(mode=f['mode'],GPU_reference=c,CPU_references=cpu_checks,original_BF16_precision_observations=precision_checks,
                        full_physical_storage_checked=True,output_and_scratch_guards=True,inputs_unchanged=True,RNG_unchanged=True,writer_fixture=writer_file))
                phases.append(dict(phase=phase,lengths=list(lengths),checks=checks,changed_K_V_scales_slots_lengths=bool(phase)))
            raw_ms={mode:[] for mode in MODES}
            for trial in range(7):
                order=fixtures[trial%len(MODES):]+fixtures[:trial%len(MODES)]
                for f in order:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
                    for _ in range(20):f['graph'].replay()
                    end.record();end.synchronize();raw_ms[f['mode']].append(start.elapsed_time(end)/20)
            for f in fixtures:assert torch.equal(f['storage'],f['expected_storage'])
            medians={mode:statistics.median(raw_ms[mode]) for mode in MODES}
            state['cases'].append(dict(batch=n,context=context,strided=bool(case_id%2),phases=phases,layouts=[f['layout'] for f in fixtures],
                raw_ms=raw_ms,median_ms=medians,segments=segments,same_page_table_for_all_modes=True,timing_includes_real_writer_quantizer_scales=True));save()
            print(json.dumps(dict(batch=n,context=context,median_ms=medians)),flush=True)
            del fixtures,f,graph,expected,keys,values,scores,answer,logical_k,logical_v,qs,q,cu,used,common_table
            gc.collect();torch.cuda.empty_cache()
        assert len(state['cases'])==5 and all(sha(path)==digest for path,digest in sources.items())
        state.update(phase='complete',complete=True);save()
    except BaseException as error:
        import traceback
        state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
