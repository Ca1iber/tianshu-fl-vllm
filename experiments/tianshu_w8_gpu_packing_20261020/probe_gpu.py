# SPDX-License-Identifier: Apache-2.0
"""All 43 real banks: exact CPU/GPU pack parity and loader-style cold timings."""
import argparse,gc,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent
COMPONENT=BASE.parent/'tianshu_dense_w8a8_20261020'
SIG=BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200;signal.alarm(min(1800,remaining))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    root=args.output.resolve();assert root.is_relative_to(Path('/workspace/benchmark-results')) and not root.exists();root.mkdir(parents=True)
    import torch,vllm_fl,flag_gems
    from safetensors import safe_open
    import packing
    torch.set_num_threads(4)
    assert getattr(flag_gems,'current_work_registrar',None) is None
    assert sha(COMPONENT/'kernel.py')=='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152'
    spec=importlib.util.spec_from_file_location('_tianshu_static_w8_pack_verified_kernel',COMPONENT/'kernel.py');kernel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernel;spec.loader.exec_module(kernel)
    paths=[BASE/name for name in ('packing.py','probe_gpu.py','verify_cpu.py','run_after_sql.sh','README.md')]+[COMPONENT/'kernel.py',SIG,Path('/workspace/MiniCPM5-2B/config.json')]
    frozen={str(p):sha(p) for p in paths}
    for p in paths:
        dst=root/'source_snapshot'/p.relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    reference=json.loads(SIG.read_text());assert len(reference)==43
    result=dict(complete=False,phase='real_banks',runner_pid=os.getpid(),frozen_sources=frozen,CPU_threads=torch.get_num_threads(),banks=[],synthetic=[],timing_rounds=[],
        original_kernel_sha256=sha(COMPONENT/'kernel.py'),source_signature=str(SIG),model_executed=False,adopted=False,official_benchmark=False,
        timing_scope='Same 4 CPU threads in both arms: BF16 D2H, finite check, packing, GPU payload allocation/transfer, and three full payload SHA hashes; candidate also copies GPU q/scales to CPU for the same proof. First candidate compile is included.',
        timing_is_not_actual_model_startup=True,no_CUDA_Graph=True,all_original_checkpoint_weights_retained=True)
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(root/'result.json')
    def tensor_sha(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
    def cpu_pack(w):
        n,k=w.shape;q=torch.empty((n,k),dtype=torch.int8);s=torch.empty(n,dtype=torch.float32)
        for lo in range(0,n,8192):
            block=w[lo:lo+8192].float();scale=block.abs().amax(1).clamp_min(1.e-8)/127.
            q[lo:lo+len(block)]=torch.round(block/scale[:,None]).clamp(-127,127).to(torch.int8);s[lo:lo+len(block)]=scale
        return q,s
    def run(weight,mode):
        torch.cuda.synchronize();started=time.perf_counter();wc=weight.detach().cpu().contiguous();assert bool(torch.isfinite(wc).all())
        n,k=wc.shape
        if mode=='native':
            qc,sc=cpu_pack(wc);q=qc.to(weight.device);s=sc.to(weight.device);qstore=sstore=None;compiled=[]
        else:
            qstore=torch.full((n*k+16,),17,device=weight.device,dtype=torch.int8);q=qstore[:n*k].view(n,k)
            sstore=torch.full((n+16,),17.,device=weight.device,dtype=torch.float32);s=sstore[:n]
            compiled=packing.pack_into(weight,q,s,kernel);qc,sc=q.cpu(),s.cpu()
        signature=dict(original_BF16_sha256=tensor_sha(wc),packed_sha256=tensor_sha(qc),scales_sha256=tensor_sha(sc))
        torch.cuda.synchronize();milliseconds=(time.perf_counter()-started)*1000
        guards=mode=='native' or bool((qstore[n*k:]==17).all() and (sstore[n:]==17).all())
        assert guards
        metadata=[dict(n_regs=getattr(c,'n_regs',None),n_spills=getattr(c,'n_spills',None)) for c in compiled]
        return q,s,qc,sc,signature,milliseconds,metadata
    save()
    try:
        files=list(Path('/workspace/MiniCPM5-2B').glob('*.safetensors'));assert len(files)==1
        with safe_open(str(files[0]),framework='pt',device='cpu') as checkpoint:
            for bank in reference:
                prefix=bank['prefix']
                if bank['role']=='gateup':
                    stem=prefix.removesuffix('gate_up_proj');keys=[stem+name+'_proj.weight' for name in ('gate','up')]
                else:assert bank['role']=='lm_head';keys=['lm_head.weight']
                wc=torch.cat([checkpoint.get_tensor(key) for key in keys],0).contiguous();assert list(wc.shape)==bank['shape']
                w=wc.cuda();version=w._version;cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                row=dict(prefix=prefix,role=bank['role'],shape=list(wc.shape),checkpoint_keys=keys,rounds=[]);result['banks'].append(row);save()
                for index in range(3):
                    order=['native','candidate'] if index%2==0 else ['candidate','native'];outputs={}
                    for mode in order:outputs[mode]=run(w,mode)
                    a,b=outputs['native'],outputs['candidate']
                    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and torch.equal(a[2],b[2]) and torch.equal(a[3],b[3])
                    assert a[4]==b[4]=={key:bank[key] for key in ('original_BF16_sha256','packed_sha256','scales_sha256')}
                    assert tensor_sha(w.cpu())==bank['original_BF16_sha256'] and w._version==version
                    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
                    row['rounds'].append(dict(round=index,full_CPU_GPU_q_and_scale_bitwise_equal=True,original_input_bitwise_unchanged=True,RNG_unchanged=True,guards_passed=True,signature=b[4],compiled=b[6]))
                    result['timing_rounds'].append(dict(prefix=prefix,round=index,order=order,native_ms=a[5],candidate_ms=b[5]));save()
                    del outputs,a,b;gc.collect()
                del w,wc;gc.collect();torch.cuda.empty_cache()
        generator=torch.Generator().manual_seed(202610202210)
        for n,k,strided in [(1,2048,False),(17,259,True),(8193,33,False)]:
            initial=torch.randn(n,k,generator=generator,dtype=torch.bfloat16);initial[0].zero_()
            if n>1:initial[1].zero_();initial[1,:9]=torch.tensor([.5,1.5,2.5,3.5,-.5,-1.5,-2.5,-3.5,127.],dtype=torch.bfloat16)
            storage=torch.full((n,k+(7 if strided else 0)),17.,device='cuda',dtype=torch.bfloat16);w=storage[:,:k]
            qstore=torch.full((n*k+16,),17,device='cuda',dtype=torch.int8);q=qstore[:n*k].view(n,k)
            sstore=torch.full((n+16,),17.,device='cuda');s=sstore[:n]
            for phase,factor in enumerate([1.,-.5,0.]):
                wc=initial*factor;w.copy_(wc);qc,sc=cpu_pack(wc);before=tensor_sha(w.cpu())
                cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                packing.pack_into(w,q,s,kernel);torch.cuda.synchronize()
                assert torch.equal(q.cpu(),qc) and torch.equal(s.cpu(),sc) and tensor_sha(w.cpu())==before
                assert bool((qstore[n*k:]==17).all() and (sstore[n:]==17).all() and (storage[:,k:]==17).all())
                assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
                filename=f'synthetic_N{n}_K{k}_phase{phase}.pt';torch.save(dict(N=n,K=k,phase=phase,input=wc,packed=q.cpu(),scales=s.cpu()),root/filename)
                result['synthetic'].append(dict(N=n,K=k,strided=strided,phase=phase,file=filename,full_bitwise_equal=True,guards_passed=True,RNG_unchanged=True,input_bitwise_unchanged=True));save()
            del storage,w,qstore,q,sstore,s;gc.collect();torch.cuda.empty_cache()
        assert len(result['banks'])==43 and len(result['synthetic'])==9 and all(sha(p)==v for p,v in frozen.items())
        result.update(complete=True,phase='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()
    except BaseException as error:result.update(phase='failed',failure=repr(error));save();raise
if __name__=='__main__':main()
