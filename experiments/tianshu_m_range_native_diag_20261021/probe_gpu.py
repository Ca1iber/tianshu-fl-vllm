# SPDX-License-Identifier: Apache-2.0
"""Diagnose actual vendor O8 MM constants/eager/graph using full saved outputs."""
import hashlib,json,signal,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261021_m_range_native_diag_GPU_tianshu')

def main():
    seconds=int((datetime.fromisoformat('2026-10-21T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert seconds>0
    signal.alarm(min(seconds,300))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['CPU_only']
    for filename,digest in gate['sources'].items():assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
    ROOT.mkdir(exist_ok=False);(ROOT/'hardware_before.txt').write_text(hardware)
    result=dict(complete=False,phase='starting',scope='diagnostic execution, not model adoption',original_BF16_range_plan_still_failed=True,cases=[],sources=gate['sources'])
    def save():(ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        import torch,flag_gems
        from safetensors import safe_open
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        native=sys.modules[flag_gems.mm_out.__module__]
        assert Path(native.__file__).resolve()==Path('/usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/ops/mm.py')
        result['native_source']=native.__file__;result['actual_launches']=[]
        original_kernel=native.mm_kernel
        class Spy:
            def __getitem__(self,grid):
                def call(*a,**kw):
                    kernel=original_kernel[grid](*a,**kw)
                    source=getattr(kernel,'src',None);constants=getattr(source,'constants',{})
                    names=getattr(getattr(source,'fn',None),'arg_names',[])
                    resolved={names[key] if isinstance(key,int) and 0<=key<len(names) else str(key):str(value) for key,value in constants.items()}
                    result['actual_launches'].append(dict(capture=torch.cuda.is_current_stream_capturing(),cache_hash=kernel.hash,
                        binary_sha256=hashlib.sha256(kernel.kernel).hexdigest(),constants=resolved,
                        warps=kernel.metadata.num_warps,stages=kernel.metadata.num_stages,
                        n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None)))
                    if 'ttir' in kernel.asm:(ROOT/'actual_native_ttir.mlir').write_text(kernel.asm['ttir'])
                    return kernel
                return call
        native.mm_kernel=Spy()
        with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as f:
            wc=f.get_tensor('model.layers.0.self_attn.o_proj.weight').contiguous()
        ac=torch.randn(8,2048,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(202610210010));ac[0].zero_()
        def compare(actual,reference):
            x,y=actual.float(),reference.float();delta=(x-y).abs();bad=delta>.02+.01*y.abs()
            return dict(passed=bool(torch.isfinite(x).all() and torch.isfinite(y).all() and not bad.any()),
                failed_elements=int(bad.sum()),max_abs=float(delta.max()),atol=.02,rtol=.01)
        with torch.inference_mode():
            a=ac.cuda();w=wc.cuda();out=torch.empty(8,2048,device='cuda',dtype=torch.bfloat16)
            def invoke():return flag_gems.mm_out(a,w.T,out=out)
            for _ in range(2):invoke()
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):invoke()
            for phase in range(2):
                factor=1. if phase==0 else -.5
                a.copy_(ac*factor);w.copy_(wc*factor)
                reference_cpu=((ac*factor).double()@(wc*factor).double().T).to(torch.bfloat16)
                payload=dict(phase=phase,original_A=ac,original_W=wc,A_expected=ac*factor,W_expected=wc*factor,
                    reference_CPU_FP64_to_BF16=reference_cpu,outputs={})
                checks={}
                for repeats in (1,20,140):
                    for _ in range(repeats):graph.replay()
                    torch.cuda.synchronize();value=out.cpu().clone();label=f'fixed_pointer_graph_{repeats}'
                    payload['outputs'][label]=value;checks[label]=compare(value,reference_cpu)
                invoke();torch.cuda.synchronize();payload['outputs']['same_pointer_eager']=out.cpu().clone()
                checks['same_pointer_eager']=compare(payload['outputs']['same_pointer_eager'],reference_cpu)
                fresh_a=(ac*factor).cuda();fresh_w=(wc*factor).cuda();fresh_out=torch.empty_like(out)
                flag_gems.mm_out(fresh_a,fresh_w.T,out=fresh_out);torch.cuda.synchronize()
                payload['outputs']['fresh_pointer_eager']=fresh_out.cpu().clone()
                checks['fresh_pointer_eager']=compare(payload['outputs']['fresh_pointer_eager'],reference_cpu)
                fresh_graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(fresh_graph):flag_gems.mm_out(fresh_a,fresh_w.T,out=fresh_out)
                fresh_graph.replay();torch.cuda.synchronize();payload['outputs']['fresh_pointer_graph']=fresh_out.cpu().clone()
                checks['fresh_pointer_graph']=compare(payload['outputs']['fresh_pointer_graph'],reference_cpu)
                payload['actual_A_after']=a.cpu();payload['actual_W_after']=w.cpu()
                torch.save(payload,ROOT/f'phase{phase}.pt')
                unchanged=bool(torch.equal(a.cpu(),ac*factor) and torch.equal(w.cpu(),wc*factor))
                result['cases'].append(dict(phase=phase,checks=checks,inputs_unchanged=unchanged))
                save();print(json.dumps(result['cases'][-1]),flush=True)
        result.update(complete=True,phase='complete',full_tensors_saved=True,numerical_tolerances_unchanged=True)
        save()
    except BaseException as error:
        result.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise

if __name__=='__main__':main()
