# SPDX-License-Identifier: Apache-2.0
"""Actual-weight INT8 GEMM probe; every timed replay includes dynamic A quantization."""
import argparse
from datetime import datetime, timezone
import gc, hashlib, inspect, json, os, signal, statistics, subprocess, time
from pathlib import Path
BASE = Path(__file__).resolve().parent
PLANS = json.loads((BASE/"tiles.json").read_text())
SHAPES = [("gateup",12288,2048)]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200
    signal.alarm(min(1800,remaining))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    root=args.output.resolve();assert root.is_relative_to(Path('/workspace/benchmark-results')) and not root.exists();root.mkdir(parents=True)
    (root/'hardware_before.txt').write_text(hardware)
    import torch, vllm_fl, flag_gems
    from safetensors import safe_open
    import kernel
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    assert getattr(flag_gems,'current_work_registrar',None) is None
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['INT32_no_overflow']
    assert all(digest(path)==sha for path,sha in gate['sources'].items())
    files=[BASE/name for name in ('kernel.py','check_cpu.py','CPU_GATE_32g.json','probe_gpu.py','run_component.sh','tiles.json')]
    files += [Path(inspect.getsourcefile(flag_gems.linear)),Path('/workspace/MiniCPM5-2B/config.json')]
    frozen={**gate["sources"],**{str(path):digest(path) for path in files}}
    for path in map(Path,frozen):
        target=root/'source_snapshot'/path.relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    result=dict(complete=False,phase='weights',runner_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),frozen_sources=frozen,
        plans=PLANS,cases=[],weights={},timing_rounds=[],official_benchmark=False,adopted=False,model_executed=False,
        timing_scope='Hot CUDA Graph: per-token absmax/RNE A quantization plus integer GEMM and dequantization; static W packing excluded and recorded.',
        baseline='flag_gems.linear(A_BF16.contiguous(), W_BF16, None), hot Graph, same actual checkpoint weights and activations',
        numerical_scope='Full exact INT32 GPU check where actual sum(abs(products))<2**24; independent selected CPU INT64; full BF16 output against quantized reference. Original BF16 approximation error separately reported.',
        full_GPU_checks_saved_as_assertions=True,offline_CPU_verifier_scope='Full A quantization and selected original weight rows / output rows and columns; not offline full GPU output verification',checkpoint_files_never_modified=True)
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(root/'result.json')
    def tensor_sha(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
    def pack(w):
        q=torch.empty(w.shape,dtype=torch.int8);scales=torch.empty(w.shape[0],dtype=torch.float32)
        for lo in range(0,len(w),8192):
            block=w[lo:lo+8192].float();scale=block.abs().amax(1).clamp_min(1.e-8)/127.
            q[lo:lo+len(block)]=torch.round(block/scale[:,None]).clamp(-127,127).to(torch.int8);scales[lo:lo+len(block)]=scale
        return q,scales
    def metadata(compiled):
        return dict(cache_hash=compiled.hash,binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),shared=getattr(getattr(compiled,'metadata',None),'shared',None))
    def measure(graph):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
        for _ in range(20):graph.replay()
        b.record();b.synchronize();return a.elapsed_time(b)/20
    save()
    try:
        paths=list(Path('/workspace/MiniCPM5-2B').glob('*.safetensors'));assert len(paths)==1
        prefix='model.layers.0.'
        keys={'qkv':[prefix+'self_attn.'+s+'_proj.weight' for s in ('q','k','v')],
            'o':[prefix+'self_attn.o_proj.weight'],'gateup':[prefix+'mlp.'+s+'_proj.weight' for s in ('gate','up')],
            'down':[prefix+'mlp.down_proj.weight'],'lm_head':['lm_head.weight']}
        generator=torch.Generator().manual_seed(202610201950)
        with safe_open(str(paths[0]),framework='pt',device='cpu') as checkpoint:
          for shape_id,(name,n,k) in enumerate(SHAPES):
            if name=='tail':
                wc=torch.randn(n,k,generator=generator,dtype=torch.bfloat16)*.02;wc[-1].zero_()
            else:wc=torch.cat([checkpoint.get_tensor(key) for key in keys[name]],0).contiguous()
            assert wc.shape==(n,k) and wc.dtype==torch.bfloat16 and bool(torch.isfinite(wc).all())
            started=time.perf_counter();wqc,wsc=pack(wc);packing=time.perf_counter()-started
            result['weights'][name]=dict(shape=[n,k],keys=keys.get(name,[]),checkpoint=str(paths[0]) if name!='tail' else None,
                BF16_bits_sha256=tensor_sha(wc),packed_sha256=tensor_sha(wqc),scales_sha256=tensor_sha(wsc),CPU_static_packing_seconds=packing,
                BF16_bytes=wc.numel()*2,INT8_and_scale_bytes=wqc.numel()+wsc.numel()*4,
                extra_bytes_if_BF16_retained_for_prefill=wqc.numel()+wsc.numel()*4)
            signature=json.loads((BASE.parent/"tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json").read_text())
            expected=next(r for r in signature if r["prefix"]=="model.layers.0.mlp.gate_up_proj")
            assert tensor_sha(wc)==expected["original_BF16_sha256"] and tensor_sha(wqc)==expected["packed_sha256"] and tensor_sha(wsc)==expected["scales_sha256"]
            for m in (4096,8192,16384):
                case=dict(name=name,M=m,N=n,K=k,strided_A=name=='tail',configurations=[],phase_files=[]);result['cases'].append(case);save()
                original_a=torch.randn(m,k,generator=generator,dtype=torch.bfloat16)
                if name=='tail':
                    original_a[-1].zero_();original_a[-2].zero_();original_a[-2,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
                astore=torch.empty((m,k+(7 if name=='tail' else 0)),device='cuda',dtype=torch.bfloat16);a=astore[:,:k];a.copy_(original_a)
                w=wc.cuda();wq_store=torch.full((n*k+16,),17,device='cuda',dtype=torch.int8);wq=wq_store[:n*k].view(k,n).T;wq.copy_(wqc)
                ws_store=torch.full((n+16,),17.,device='cuda');ws=ws_store[:n];ws.copy_(wsc)
                q_store=torch.full((m*k+16,),17,device='cuda',dtype=torch.int8);q=q_store[:m*k].view(m,k)
                asc_store=torch.full((m+16,),17.,device='cuda');asc=asc_store[:m]
                out_store=torch.full((m*n+16,),17.,device='cuda',dtype=torch.bfloat16);out=out_store[:m*n].view(m,n)
                int_store=torch.full((m*n+16,),-777,device='cuda',dtype=torch.int32);integer=int_store[:m*n].view(m,n)
                rows=torch.linspace(0,m-1,min(8,m)).long();columns=torch.linspace(0,n-1,min(32,n)).long();rdev,cdev=rows.cuda(),columns.cuda()
                archives=[]
                for phase in range(3):
                    ac=original_a if phase==0 else original_a*(.5 if phase==1 else -.5)
                    sign=1 if phase==0 else -1;scale_factor=.5 if phase==2 else 1.
                    aq,as_cpu=pack(ac)
                    archives.append(dict(phase=phase,a=ac,rows=rows,columns=columns,original_weight_rows=wc[columns].clone(),
                        actual_A_INT8=aq,actual_A_scale=as_cpu,actual_W_INT8=wqc[columns]*sign,actual_W_scale=wsc[columns]*scale_factor,
                        CPU_INT64=(aq[rows].to(torch.int64)@(wqc[columns]*sign).to(torch.int64).T),outputs={}))
                    case['phase_files'].append(f'{name}_M{m}_phase{phase}.pt')
                for index,config in enumerate(PLANS):
                    entry=dict(index=index,configuration=config,phases=[]);case['configurations'].append(entry);save()
                    a.copy_(original_a);w.copy_(wc);wq.copy_(wqc);ws.copy_(wsc)
                    graphs={};native_out=None
                    try:
                        for _ in range(3):
                            native_out=flag_gems.linear(a.contiguous(),w,None);quant_kernel=kernel.quantize(a,q,asc)
                            gemm_kernel=kernel.mm(q,wq.T,asc,ws,out,integer,config)
                        torch.cuda.synchronize()
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):native_out=flag_gems.linear(a.contiguous(),w,None)
                        graphs['native']=graph
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):kernel.quantize(a,q,asc);kernel.mm(q,wq.T,asc,ws,out,integer,config)
                        graphs['candidate']=graph
                    except Exception as error:
                        entry['compile_error']=repr(error);save()
                        if any(s in str(error).lower() for s in ('illegal memory','device-side assert')):raise
                        graphs.clear();gc.collect();continue
                    entry['compiled']=True;entry['gemm_metadata']=metadata(gemm_kernel);entry['quantize_metadata']=metadata(quant_kernel)
                    for phase,archive in enumerate(archives):
                        a.copy_(archive['a']);sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.
                        w.copy_(wc*sign*factor);wq.copy_(wqc*sign);ws.copy_(wsc*factor)
                        x_before=a.clone();w_before=w.clone();qweight_before=wq.clone();scale_before=ws.clone()
                        rng=torch.cuda.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone()
                        graphs['native'].replay();graphs['candidate'].replay();torch.cuda.synchronize()
                        assert torch.equal(q.cpu(),archive['actual_A_INT8']) and torch.equal(asc.cpu(),archive['actual_A_scale'])
                        bound=q.float().abs()@wq.float().abs().T;max_sum_abs=float(bound.max());assert max_sum_abs<2**24
                        integer_reference=q.float()@wq.float().T
                        kernel.mm(q,wq.T,asc,ws,out,integer,config,return_int=True);torch.cuda.synchronize()
                        assert torch.equal(integer.float(),integer_reference)
                        assert torch.equal(integer[rdev[:,None],cdev[None,:]].cpu().to(torch.int64),archive['CPU_INT64'])
                        graphs['candidate'].replay();torch.cuda.synchronize()
                        reference=integer_reference*asc[:,None]*ws[None,:]
                        error=(out.float()-reference).abs();assert bool(torch.isfinite(out).all()) and not bool((error>(.02+.01*reference.abs())).any())
                        native_selected=native_out[rdev[:,None],cdev[None,:]].cpu()
                        native_reference=archive['a'][rows].float()@(archive['original_weight_rows'].float()*sign*factor).T
                        assert not bool(((native_selected.float()-native_reference).abs()>(.02+.01*native_reference.abs())).any())
                        assert torch.equal(a,x_before) and torch.equal(w,w_before) and torch.equal(wq,qweight_before) and torch.equal(ws,scale_before)
                        assert torch.equal(torch.cuda.get_rng_state(),rng) and torch.equal(torch.get_rng_state(),cpu_rng)
                        assert bool((q_store[m*k:]==17).all() and (asc_store[m:]==17).all() and (out_store[m*n:]==17).all() and (int_store[m*n:]==-777).all() and (wq_store[n*k:]==17).all() and (ws_store[n:]==17).all())
                        delta=out.float()-native_out.float()
                        observed=dict(phase=phase,full_integer_exact=True,full_quantized_output_passed=True,max_abs_quantized_error=float(error.max()),
                            actual_max_sum_abs_integer_products=max_sum_abs,CPU_INT64_selected_exact=True,guards_passed=True,inputs_unchanged=True,RNG_unchanged=True,
                            original_BF16_relative_L2=float((delta.square().sum()/native_out.float().square().sum().clamp_min(1.e-20)).sqrt()),
                            original_BF16_max_abs=float(delta.abs().max()),original_BF16_top1_agreement=float((out.argmax(1)==native_out.argmax(1)).float().mean()))
                        entry['phases'].append(observed)
                        archive['outputs'][str(index)]=dict(integer=integer[rdev[:,None],cdev[None,:]].cpu(),candidate=out[rdev[:,None],cdev[None,:]].cpu(),native=native_selected)
                        torch.save(archive,root/case['phase_files'][phase])
                        del bound,integer_reference,reference,error,delta,x_before,w_before,qweight_before,scale_before
                    for round_index in range(7):
                        order=['native','candidate'] if round_index%2==0 else ['candidate','native'];times={mode:measure(graphs[mode]) for mode in order}
                        result['timing_rounds'].append(dict(case=name,M=m,index=index,round=round_index,order=order,replays=20,**{mode+'_ms':value for mode,value in times.items()}))
                    assert bool((q_store[m*k:]==17).all() and (asc_store[m:]==17).all() and (out_store[m*n:]==17).all() and (int_store[m*n:]==-777).all())
                    entry['timing_guards_passed']=True;save()
                    print(json.dumps(dict(case=name,M=m,index=index,compiled=True,phases=3)),flush=True)
                    graphs.clear();del graphs,graph,native_out;gc.collect()
                assert any(c.get('compiled') for c in case['configurations'])
                for archive,path in zip(archives,case['phase_files'],strict=True):torch.save(archive,root/path)
                del a,astore,w,wq_store,wq,ws_store,ws,q_store,q,asc_store,asc,out_store,out,int_store,integer,archives
                gc.collect();torch.cuda.empty_cache();save()
            del wc,wqc,wsc
        assert len(result['cases'])==3
        assert all(digest(path)==sha for path,sha in frozen.items())
        result.update(complete=True,phase='complete',finished_utc=datetime.now(timezone.utc).isoformat(),model_accuracy_not_evaluated=True)
        save()
    except BaseException as error:
        result.update(phase='failed',failure=repr(error));save();raise
if __name__=='__main__':main()
