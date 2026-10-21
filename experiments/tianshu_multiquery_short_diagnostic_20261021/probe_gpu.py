# SPDX-License-Identifier: Apache-2.0
"""Short-key numerical cause diagnostic, recording ideal FP32 failures honestly."""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'tianshu_multiquery_splitkv_20261021'
sys.path.insert(0,str(OLD))
from source import sha,load,PARENT,HELPERS,CONFIGS,segments
import launch

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
    seconds=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert seconds>300;signal.alarm(min(900,seconds))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    root=args.output;assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();(root/'hardware_before.txt').write_text(hardware)
    import torch,vllm_fl
    from emulate import emulate
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    spec=importlib.util.spec_from_file_location('_shortdiag_value_planes',BASE.parent/'tianshu_k8v16_pipeline_20261019/value_planes.py');planes=importlib.util.module_from_spec(spec);spec.loader.exec_module(planes)
    cpu=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert cpu['complete'] and cpu['CPU_only']
    frozen={**cpu['sources'],str(BASE/'CPU_GATE_32g.json'):sha(BASE/'CPU_GATE_32g.json')};models=[]
    for label,c in [('control',None),('old_candidate0',CONFIGS[0]),('Q2_T16_w1',dict(block_m=16,tile=16,warps=1,stages=1)),('Q4_T16_w2',dict(block_m=32,tile=16,warps=2,stages=1))]:
        mode='control' if c is None else 'candidate';reader,proof=load(root/label,mode);producer,reducer=launch.attach(reader,mode,c)
        models.append(dict(label=label,config=c,reader=reader,producer=producer,proof=proof,split=mode=='candidate'));frozen[reader.__file__]=sha(reader.__file__)
    for path,h in frozen.items():
        assert sha(path)==h;dest=root/'source_snapshot'/Path(path).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(path).read_bytes())
    result=dict(complete=False,phase='diagnostic',runner_pid=os.getpid(),frozen_sources=frozen,cases=[],
        model_executed=False,benchmark=False,ideal_FP32_failures_not_suppressed=True,CPU_emulation_has_BF16_P_cast_as_native_source=True,
        original_failed_component_preserved=True,no_tolerance_relaxation=True,old_original_FP32_threshold=dict(atol=.002,rtol=.01))
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(root/'result.json')
    def error(a,b):
        d=(a.float()-b.float()).abs();bad=d>(.002+.01*b.float().abs());return dict(passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),max_abs=float(d.max()),failed=int(bad.sum()))
    save()
    try:
      with torch.inference_mode():
        for case_id,(ql,kl,causal) in enumerate([(2,2,True),(3,17,True),(4,33,True),(8,8,True),(8,65,True),(7,17,False)]):
            n=8;width=(kl+15)//16;gen=torch.Generator().manual_seed(79000+case_id)
            qc=torch.randn(n*ql,16,128,generator=gen,dtype=torch.bfloat16);kc=torch.randn(n,width*16,2,128,generator=gen,dtype=torch.bfloat16);vc=torch.randn(n,width*16,2,128,generator=gen,dtype=torch.bfloat16)
            bits=kc.view(torch.int16).int();magnitude=bits&32767;r=((magnitude+7+((magnitude>>4)&1))>>4)-928;encoded=(torch.where(r>=8,r.clamp(max=127),0)|((bits&32768)>>8)).to(torch.uint8);mag=encoded.int()&127
            decoded=(torch.where(mag>=8,(mag<<4)+14848,0)|((encoded.int()&128)<<8)).to(torch.int16).view(torch.bfloat16)
            # CPU FP32 ideal references, and independent CPU emulation of native BF16 P semantics.
            ideals=[]
            for seq in range(n):
                score=(qc[seq*ql:(seq+1)*ql].float().transpose(0,1)@decoded[seq,:kl].float().repeat_interleave(8,1).permute(1,2,0))*(128**-.5)
                if causal:score.masked_fill_(torch.arange(kl)[None,None,:]>kl-ql+torch.arange(ql)[None,:,None],-float('inf'))
                ideals.append((score.softmax(-1)@vc[seq,:kl].float().repeat_interleave(8,1).transpose(0,1)).transpose(0,1))
            ideal=torch.cat(ideals);raw=torch.full(((n*width+2)*12288,),17,device='cuda',dtype=torch.uint8)
            kv=torch.as_strided(raw,(n*width+2,2,16,2,128),(12288,4096,256,128,1));k,v=kv.unbind(1);v=planes.value_plane(v)
            k[:n*width].copy_(encoded.reshape(n*width,16,2,128));v[:n*width].copy_(vc.reshape(n*width,16,2,128))
            table_store=torch.full((n,width+9),-1234567,device='cuda',dtype=torch.int32);table=table_store[:,:width];table.copy_(torch.arange(n*width,device='cuda',dtype=torch.int32).reshape(n,width))
            q=qc.cuda();used=torch.full((n,),kl,device='cuda',dtype=torch.int32);cu=torch.arange(n+1,device='cuda',dtype=torch.int32)*ql;s=segments(n)
            row=dict(query_length=ql,key_length=kl,causal=causal,modes=[]);outputs={};result['cases'].append(row);save()
            raw_before=raw.clone();q_before=q.clone();used_before=used.clone();cu_before=cu.clone();table_before=table_store.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
            for model in models:
                config=model['config'] or dict(block_m=256,tile=128,warps=16,stages=1)
                emulated=emulate(qc,decoded,vc,[ql]*n,[kl]*n,config,model['split'],s,causal)
                out_store=torch.full((n*ql+1,16,128),17.,device='cuda',dtype=torch.bfloat16);out=out_store[:-1]
                scratch=torch.full((n*ql+1,16,s,128),17.,device='cuda',dtype=torch.float32);maximum=torch.full((n*ql+1,16,s),17.,device='cuda',dtype=torch.float32);expsum=torch.full_like(maximum,17.)
                kw=dict(q=q,k=k,v=v,out=out,cu_seqlens_q=cu,max_seqlen_q=ql,seqused_k=used,max_seqlen_k=width*16,softmax_scale=128**-.5,
                    causal=causal,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=None,k_descale=None,v_descale=None,
                    seq_threshold_3D=64,num_par_softmax_segments=s,softmax_segm_output=scratch,softmax_segm_max=maximum,softmax_segm_expsum=expsum)
                model['reader'].unified_attention(**kw);torch.cuda.synchronize();actual=out.cpu();outputs[model['label']]=actual
                emulation_error=error(actual,emulated);assert emulation_error['passed'],(ql,kl,model['label'],emulation_error)
                assert torch.equal(raw,raw_before) and torch.equal(q,q_before) and torch.equal(used,used_before) and torch.equal(cu,cu_before) and torch.equal(table_store,table_before)
                assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
                assert bool((out_store[-1]==17).all() and (scratch[-1]==17).all() and (maximum[-1]==17).all() and (expsum[-1]==17).all())
                row['modes'].append(dict(label=model['label'],config=config,ideal_FP32_error=error(actual,ideal),CPU_BF16_P_emulation_error=emulation_error,
                    emulated_vs_ideal=error(emulated,ideal),bitwise_vs_control=torch.equal(actual,outputs['control']),cache_and_guards_RNG_preserved=True,actual_launch=model['producer'].last));save()
            torch.save(dict(q=qc,original_K=kc,decoded_K=decoded,V=vc,ideal=ideal,outputs=outputs,row=row),root/f'short_{case_id}.pt')
            row['archive_sha256']=sha(root/f'short_{case_id}.pt');save();print(json.dumps(row),flush=True)
        assert all(sha(path)==h for path,h in frozen.items())
        result.update(complete=True,phase='complete',all_24_modes_match_CPU_native_BF16_P_semantics=True);save()
    except BaseException as exc:result.update(phase='failed',error=repr(exc));save();raise
if __name__=='__main__':main()
