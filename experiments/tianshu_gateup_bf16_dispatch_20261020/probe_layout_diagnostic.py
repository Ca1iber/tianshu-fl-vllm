# SPDX-License-Identifier: Apache-2.0
"""Preserve numerical failures; isolate layout and measure two dominant prefill sizes."""
import argparse,gc,hashlib,inspect,json,os,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent
CASES=[(565,False),(565,True),(1537,False),(1537,True),(2048,False),(4096,False)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200;signal.alarm(min(1800,remaining))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    root=args.output.resolve();assert root.is_relative_to(Path('/workspace/benchmark-results')) and not root.exists();root.mkdir(parents=True)
    import torch,vllm_fl,flag_gems
    from safetensors import safe_open
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    assert getattr(flag_gems,'current_work_registrar',None) is None
    paths=[BASE/name for name in ('probe_layout_diagnostic.py','verify_layout_cpu.py','run_layout_diagnostic.sh','LAYOUT_DIAGNOSTIC.md')]
    paths += [Path(inspect.getsourcefile(flag_gems.mm)),Path(inspect.getsourcefile(flag_gems.linear)),Path('/workspace/FlagGems/src/flag_gems/utils/models/sql.py')]
    frozen={str(p):sha(p) for p in paths}
    for p in paths:
        dst=root/'source_snapshot'/p.relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    result=dict(complete=False,phase='loading',runner_pid=os.getpid(),frozen_sources=frozen,cases=[],timing_rounds=[],model_executed=False,adopted=False,official_benchmark=False,
        original_failing_root='/workspace/benchmark-results/20261020_gateup_bf16_dispatch_GPU_tianshu',
        numerical_tolerance=dict(atol=.02,rtol=.01),offline_CPU_scope='Archived regular selections plus actual violating pairs; not full offline GPU outputs',
        methods={name:dict(module=fn.__module__,source=inspect.getsourcefile(fn)) for name,fn in [('mm',flag_gems.mm),('linear',flag_gems.linear)]})
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
        with safe_open(str(files[0]),framework='pt',device='cpu') as checkpoint:wc=torch.cat([checkpoint.get_tensor(key) for key in keys],0).contiguous()
        n,k=wc.shape;assert (n,k)==(12288,2048)
        result['weight']=dict(checkpoint=str(files[0]),keys=keys,BF16_bits_sha256=tensor_sha(wc));generator=torch.Generator().manual_seed(202610202130)
        for m,strided in CASES:
            case_id=f'M{m}_s{int(strided)}';case=dict(case_id=case_id,M=m,N=n,K=k,strided_A=strided,phases=[],phase_files=[]);result['cases'].append(case);save()
            ac=torch.randn(m,k,generator=generator,dtype=torch.bfloat16);ac[-1].zero_()
            storage=torch.full((m,k+(7 if strided else 0)),17.,device='cuda',dtype=torch.bfloat16);a=storage[:,:k];a.copy_(ac)
            wstore=torch.full((n*k+16,),17.,device='cuda',dtype=torch.bfloat16);w=wstore[:n*k].view(n,k);w.copy_(wc)
            graphs={};outputs={}
            for name,fn in [('mm',lambda:flag_gems.mm(a,w.T)),('linear',lambda:flag_gems.linear(a,w,None))]:
                for _ in range(3):outputs[name]=fn()
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs[name]=fn()
                graphs[name]=graph
            rows=torch.linspace(0,m-1,8).long();cols=torch.linspace(0,n-1,32).long();rd,cd=rows.cuda(),cols.cuda()
            for phase in range(3):
                actual_a=ac*[1.,.5,-.5][phase];actual_w=wc*(1. if phase==0 else -1.);a.copy_(actual_a);w.copy_(actual_w)
                rng=torch.cuda.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone();input_sha=tensor_sha(a.cpu());weight_sha=tensor_sha(w.cpu())
                for graph in graphs.values():graph.replay()
                torch.cuda.synchronize();ref=a.float()@w.float().T
                archive=dict(phase=phase,case_id=case_id,rows=rows,columns=cols,input_rows=actual_a[rows],weight_rows=actual_w[cols],outputs={},failures={})
                checks={}
                for name,out in outputs.items():
                    diff=(out.float()-ref).abs();bad=(~torch.isfinite(out))|(diff>(.02+.01*ref.abs()));count=int(bad.sum());passed=count==0
                    checks[name]=dict(full_GPU_reference_passed=passed,violating_elements=count,nonfinite_elements=int((~torch.isfinite(out)).sum()),max_abs_error=float(diff.max()))
                    archive['outputs'][name]=out[rd[:,None],cd[None,:]].cpu()
                    if not passed:
                        positions=bad.nonzero()[:32].cpu();fr,fc=positions[:,0],positions[:,1];fdev,cdev=fr.cuda(),fc.cuda()
                        cpu_reference=(actual_a[fr].float()*actual_w[fc].float()).sum(1);gpu_reference=ref[fdev,cdev].cpu()
                        assert not bool(((cpu_reference-gpu_reference).abs()>(.001+.001*cpu_reference.abs())).any())
                        archive['failures'][name]=dict(rows=fr,columns=fc,input_rows=actual_a[fr],weight_rows=actual_w[fc],actual=out[fdev,cdev].cpu(),GPU_reference=gpu_reference,CPU_reference=cpu_reference)
                assert tensor_sha(a.cpu())==input_sha and tensor_sha(w.cpu())==weight_sha
                assert bool((storage[:,k:]==17).all() and (wstore[n*k:]==17).all())
                assert torch.equal(rng,torch.cuda.get_rng_state()) and torch.equal(cpu_rng,torch.get_rng_state())
                path=f'{case_id}_phase{phase}.pt';torch.save(archive,root/path)
                case['phase_files'].append(path);case['phases'].append(dict(phase=phase,checks=checks,inputs_bitwise_unchanged=True,guards_passed=True,RNG_unchanged=True));save()
                del diff,bad,ref
            case['both_methods_passed_all_phases']=all(p['checks'][name]['full_GPU_reference_passed'] for p in case['phases'] for name in ('mm','linear'))
            if case['both_methods_passed_all_phases']:
                for index in range(9):
                    order=['mm','linear'] if index%2==0 else ['linear','mm'];times={name:measure(graphs[name]) for name in order}
                    result['timing_rounds'].append(dict(case_id=case_id,M=m,strided_A=strided,round=index,order=order,replays=20,**{name+'_ms':v for name,v in times.items()}))
            save();print(json.dumps(dict(case_id=case_id,passed=case['both_methods_passed_all_phases'],checks=case['phases'][0]['checks'])),flush=True)
            del graphs,graph,outputs,a,storage,w,wstore;gc.collect();torch.cuda.empty_cache()
        assert len(result['cases'])==6 and all(sha(p)==v for p,v in frozen.items())
        result.update(complete=True,phase='complete');save()
    except BaseException as error:result.update(phase='failed',failure=repr(error));save();raise
if __name__=='__main__':main()
