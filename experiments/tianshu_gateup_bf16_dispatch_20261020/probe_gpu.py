# SPDX-License-Identifier: Apache-2.0
"""Actual GateUp weights: hot Graph comparison of BF16 mm and BF16 linear."""
import argparse,gc,hashlib,inspect,json,os,signal,statistics,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent
MS=[32,40,48,56,128,512,565,576,1024,1537,1664,2048,4096]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200
    signal.alarm(min(1800,remaining))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    root=args.output.resolve();assert root.is_relative_to(Path('/workspace/benchmark-results')) and not root.exists();root.mkdir(parents=True)
    import torch,vllm_fl,flag_gems
    from safetensors import safe_open
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    assert getattr(flag_gems,'current_work_registrar',None) is None
    paths=[BASE/name for name in ('probe_gpu.py','verify_gpu.py','run_component.sh','run_after_validation.sh','README.md')]
    paths += [Path(inspect.getsourcefile(flag_gems.mm)),Path(inspect.getsourcefile(flag_gems.linear)),Path('/workspace/MiniCPM5-2B/config.json')]
    frozen={str(p):sha(p) for p in paths}
    for p in paths:
        dst=root/'source_snapshot'/p.relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    (root/'hardware_before.txt').write_text(hardware)
    result=dict(complete=False,phase='loading',runner_pid=os.getpid(),frozen_sources=frozen,cases=[],timing_rounds=[],
        checkpoint_never_modified=True,model_accuracy_not_evaluated=True,official_benchmark=False,adopted=False,
        methods={name:dict(module=fn.__module__,source=inspect.getsourcefile(fn)) for name,fn in [('mm',flag_gems.mm),('linear',flag_gems.linear)]},
        timing_scope='Same static actual BF16 GateUp weights, hot CUDA Graph, API allocation/capture excluded; no INT8 packing, no service',
        numerical_scope='All outputs checked on GPU against TF32-disabled FP32 matmul; independent CPU verification covers archived selected rows/columns only',
        output_guard_available=False,input_guards_checked=True)
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(root/'result.json')
    def tensor_sha(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
    def measure(graph):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
        for _ in range(20):graph.replay()
        end.record();end.synchronize();return start.elapsed_time(end)/20
    save()
    try:
        files=list(Path('/workspace/MiniCPM5-2B').glob('*.safetensors'));assert len(files)==1
        keys=['model.layers.0.mlp.'+name+'_proj.weight' for name in ('gate','up')]
        with safe_open(str(files[0]),framework='pt',device='cpu') as checkpoint:wc=torch.cat([checkpoint.get_tensor(k) for k in keys],0).contiguous()
        n,k=wc.shape;assert (n,k)==(12288,2048) and wc.dtype==torch.bfloat16
        result['weight']=dict(checkpoint=str(files[0]),keys=keys,shape=[n,k],BF16_bits_sha256=tensor_sha(wc))
        generator=torch.Generator().manual_seed(202610202130)
        for m in MS:
            case=dict(M=m,N=n,K=k,strided_A=m in (565,1537),phase_files=[],phases=[]);result['cases'].append(case);result['phase']='correctness';save()
            ac=torch.randn(m,k,generator=generator,dtype=torch.bfloat16);ac[-1].zero_()
            astore=torch.full((m,k+(7 if case['strided_A'] else 0)),17.,device='cuda',dtype=torch.bfloat16);a=astore[:,:k];a.copy_(ac)
            wstore=torch.full((n*k+16,),17.,device='cuda',dtype=torch.bfloat16);w=wstore[:n*k].view(n,k);w.copy_(wc)
            graphs={};outputs={}
            for name,fn in [('mm',lambda:flag_gems.mm(a,w.T)),('linear',lambda:flag_gems.linear(a,w,None))]:
                for _ in range(3):outputs[name]=fn()
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs[name]=fn()
                graphs[name]=graph
            rows=torch.linspace(0,m-1,min(8,m)).long();cols=torch.linspace(0,n-1,32).long();rd,cd=rows.cuda(),cols.cuda()
            for phase in range(3):
                factor=[1.,.5,-.5][phase];sign=1. if phase==0 else -1.
                actual_a=ac*factor;actual_w=wc*sign;a.copy_(actual_a);w.copy_(actual_w)
                abefore=a.clone();wbefore=w.clone();rng=torch.cuda.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone()
                for graph in graphs.values():graph.replay()
                torch.cuda.synchronize();ref=a.float()@w.float().T
                selected={};checks={}
                for name,out in outputs.items():
                    diff=(out.float()-ref).abs();assert bool(torch.isfinite(out).all()) and not bool((diff>(.02+.01*ref.abs())).any())
                    selected[name]=out[rd[:,None],cd[None,:]].cpu();checks[name]=dict(full_GPU_reference_passed=True,max_abs_error=float(diff.max()))
                assert torch.equal(a,abefore) and torch.equal(w,wbefore)
                assert bool((wstore[n*k:]==17).all()) and bool((astore[:,k:]==17).all())
                assert torch.equal(rng,torch.cuda.get_rng_state()) and torch.equal(cpu_rng,torch.get_rng_state())
                path=f'gateup_M{m}_phase{phase}.pt'
                torch.save(dict(phase=phase,M=m,rows=rows,columns=cols,input_rows=actual_a[rows],original_weight_rows=wc[cols],weight_sign=sign,outputs=selected),root/path)
                case['phase_files'].append(path);case['phases'].append(dict(phase=phase,checks=checks,inputs_unchanged=True,input_guards_passed=True,RNG_unchanged=True,
                    bitwise_same=bool(torch.equal(outputs['mm'],outputs['linear'])),cross_method_max_abs=float((outputs['mm'].float()-outputs['linear'].float()).abs().max())))
                del ref,diff,abefore,wbefore
            result['phase']='timing';save()
            for index in range(9):
                order=['mm','linear'] if index%2==0 else ['linear','mm'];times={name:measure(graphs[name]) for name in order}
                result['timing_rounds'].append(dict(M=m,round=index,order=order,replays=20,**{name+'_ms':value for name,value in times.items()}))
            assert bool((wstore[n*k:]==17).all()) and bool((astore[:,k:]==17).all());case['timing_input_guards_passed']=True
            save();print(json.dumps(dict(M=m,phases=3,timing_groups=9)),flush=True)
            del graphs,graph,outputs,a,astore,w,wstore;gc.collect();torch.cuda.empty_cache()
        assert len(result['cases'])==len(MS) and all(sha(p)==s for p,s in frozen.items())
        result.update(complete=True,phase='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()
    except BaseException as error:result.update(phase='failed',failure=repr(error));save();raise
if __name__=='__main__':main()
