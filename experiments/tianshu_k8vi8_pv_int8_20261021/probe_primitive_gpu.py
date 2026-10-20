# SPDX-License-Identifier: Apache-2.0
"""Actual GPU primitive, both diagnostic modes, changed-input graph replay."""
import argparse,hashlib,importlib.util,json,signal,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
    seconds=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert seconds>600
    state=dict(complete=False,phase='preparing',cases=[],adopted=False,full_attention_not_run=True,Math_not_run=True,performance_not_measured=True)
    def save():
        path=root/'result.tmp';path.write_text(json.dumps(state,indent=2)+'\n');path.replace(root/'result.json')
    def stop(*args):raise TimeoutError('GPU primitive deadline/termination')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(min(seconds,900));save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
        cpu=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert cpu['complete'] and cpu['no_GPU_started']
        sources=dict(cpu['sources']);sources[str(BASE/'CPU_GATE_32g.json')]=sha(BASE/'CPU_GATE_32g.json')
        for path,digest in sources.items():
            assert sha(path)==digest
            target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==digest
        state.update(frozen_sources=sources,phase='probing');save()
        import torch
        torch.set_num_threads(8)
        spec=importlib.util.spec_from_file_location('_PV_unit_actual_kernel',BASE/'kernel_pv.py');kernel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernel;spec.loader.exec_module(kernel)
        specs=((16,32),(16,64),(16,128),(32,32),(32,64))
        for case_id,(m,t) in enumerate(specs):
            d=128;generator=torch.Generator(device='cpu').manual_seed(202610211000+case_id)
            p0=torch.rand((m,t),generator=generator,dtype=torch.float32);p0[-1].zero_()
            v0=torch.randint(-127,128,(t,d),generator=generator,dtype=torch.int64).to(torch.int8)
            s0=torch.linspace(.0001,.03,t,dtype=torch.float32)
            p_gpu=torch.full((m+3,t),17.,device='cuda');v_gpu=torch.full((t+3,d),17,device='cuda',dtype=torch.int8);s_gpu=torch.full((t+3,),17.,device='cuda')
            buffers={};graphs={};compiled={}
            def launch(diag):
                b=buffers[diag]
                return kernel.probe_primitive[(1,)](p_gpu,v_gpu,s_gpu,b['out'],b['integer'],b['quantized'],b['scales'],m,t,d,diag,num_warps=2,num_stages=1)
            for diag in (False,True):
                buffers[diag]=dict(out=torch.full((m+3,d),17.,device='cuda'),integer=torch.full((m+3,d),17,device='cuda',dtype=torch.int32),quantized=torch.full((m+3,t),17,device='cuda',dtype=torch.int8),scales=torch.full((m+3,),17.,device='cuda'))
            p_gpu[:m].copy_(p0);v_gpu[:t].copy_(v0);s_gpu[:t].copy_(s0)
            for diag in (False,True):
                value=launch(diag);torch.cuda.synchronize()
                compiled[str(diag)]=dict(cache_hash=value.hash,binary_sha256=hashlib.sha256(value.kernel).hexdigest(),n_regs=getattr(value,'n_regs',None),n_spills=getattr(value,'n_spills',None),warps=value.metadata.num_warps,shared=getattr(value.metadata,'shared',None))
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):launch(diag)
                graphs[diag]=graph
            case=dict(M=m,T=t,D=d,warps=2,stages=1,compiled=compiled,phases=[]);state['cases'].append(case);save()
            for phase,factor in enumerate((1.,.5,1.e-12)):
                p=(p0.roll(phase,1)*factor).contiguous();v=(v0 if phase%2==0 else -v0).contiguous();s=(s0*(1.+phase*.25)).contiguous()
                p_gpu[:m].copy_(p);v_gpu[:t].copy_(v);s_gpu[:t].copy_(s)
                for b in buffers.values():
                    for tensor in b.values():tensor.fill_(17)
                input_before=[x.clone() for x in (p_gpu,v_gpu,s_gpu)];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                for diag in (False,True):graphs[diag].replay()
                torch.cuda.synchronize()
                assert all(torch.equal(a,b) for a,b in zip((p_gpu,v_gpu,s_gpu),input_before))
                assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
                weighted=p*s;scale=weighted.abs().amax(1).clamp_min(1.e-30)/127.
                quantized=(weighted/scale[:,None]).round().clamp(0,127).to(torch.int8);integer=quantized.long()@v.long();expected=integer.float()*scale[:,None]
                actual={str(diag):{name:tensor.cpu() for name,tensor in b.items()} for diag,b in buffers.items()}
                for diag in (False,True):
                    b=actual[str(diag)];assert torch.equal(b['out'][:m].view(torch.int32),expected.view(torch.int32))
                    assert bool((b['out'][m:]==17).all())
                    if diag:
                        assert torch.equal(b['integer'][:m].long(),integer) and torch.equal(b['quantized'][:m],quantized)
                        assert torch.equal(b['scales'][:m].view(torch.int32),scale.view(torch.int32))
                        assert all(bool((b[name][m:]==17).all()) for name in ('integer','quantized','scales'))
                    else:assert all(bool((b[name]==17).all()) for name in ('integer','quantized','scales'))
                payload=dict(p=p,v=v,value_scales=s,actual=actual,inputs_after=[x.cpu() for x in (p_gpu,v_gpu,s_gpu)],CPU_RNG_before=cpu_rng,CPU_RNG_after=torch.get_rng_state(),GPU_RNG_before=gpu_rng,GPU_RNG_after=torch.cuda.get_rng_state())
                filename=f'case{case_id}_phase{phase}.pt';torch.save(payload,root/filename)
                row=dict(phase=phase,file=filename,sha256=sha(root/filename),full_quantization_and_scale_bitwise=True,full_INT32_matches_CPU_INT64=True,both_actual_outputs_bitwise_CPU_FP32=True,guards_inputs_and_RNG_unchanged=True)
                case['phases'].append(row);save();print(json.dumps(dict(M=m,T=t,phase=phase,passed=True)),flush=True)
            del graphs,buffers,p_gpu,v_gpu,s_gpu;torch.cuda.empty_cache()
        for path,digest in sources.items():assert sha(path)==digest
        state.update(complete=True,phase='complete',checked_configurations=len(specs),changed_input_phases=3*len(specs),both_diagnostic_modes_checked=True);save()
    except Exception as error:
        state.update(phase='failed',error=repr(error));(root/'traceback.txt').write_text(traceback.format_exc());save();raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
