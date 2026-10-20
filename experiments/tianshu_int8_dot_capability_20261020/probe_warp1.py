# SPDX-License-Identifier: Apache-2.0
"""No model/performance claim: establish whether the real backend compiles INT8 dot."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import signal,subprocess,traceback
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds())
assert remaining>0;signal.alarm(min(300,remaining))
root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results')
root.mkdir(parents=True,exist_ok=False)
hardware=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_before.txt').write_text(hardware)
assert 'No running processes found' in hardware
result=dict(complete=False,component_only=True,adopted=False,official_benchmark=False,model_executed=False,
            no_latency_or_throughput_claim=True,cases=[],frozen_sources={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (Path(__file__),BASE/'kernel.py')})
def save():(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
try:
    import torch
    import vllm_fl
    from vllm.platforms import current_platform
    from triton.runtime import driver
    import kernel
    assert current_platform.vendor_name=='iluvatar'
    prop=torch.cuda.get_device_properties(0)
    result['device']=dict(name=prop.name,total_memory=prop.total_memory,multi_processor_count=prop.multi_processor_count,
                          major=prop.major,minor=prop.minor,torch_version=torch.__version__)
    try:result['triton_device_properties']=driver.active.utils.get_device_properties(0)
    except Exception as error:result['properties_query_error']=repr(error)
    for width in (16,32):
        torch.manual_seed(2026102000+width)
        q=torch.randn((16,128),dtype=torch.bfloat16);q[0]=0
        k=torch.randint(-128,128,(128,width),dtype=torch.int8)
        scales=torch.linspace(.005,.1,width,dtype=torch.float32)
        qg,kg,sg=q.cuda(),k.cuda(),scales.cuda()
        q8=torch.empty_like(qg,dtype=torch.int8);qs=torch.empty(16,device='cuda')
        integer=torch.empty((16,width),device='cuda',dtype=torch.int32)
        scaled=torch.empty_like(integer,dtype=torch.float32)
        row=dict(width=width,phases=[])
        try:
            compiled=kernel.qk[(1,)](qg,kg,sg,q8,qs,integer,scaled,width,num_warps=1,num_stages=1)
            torch.cuda.synchronize()
        except Exception as error:
            row.update(compilation_or_initial_launch_failed=True,error=repr(error),traceback=traceback.format_exc(),
                       hardware_support_not_determined_by_failure=True)
            result['cases'].append(row);save()
            if 'illegal memory' in str(error).lower() or 'device-side assert' in str(error).lower():raise
            continue
        row['compiled_metadata']=dict(n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),
                                      shared=getattr(compiled.metadata,'shared',None),num_warps=compiled.metadata.num_warps)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):kernel.qk[(1,)](qg,kg,sg,q8,qs,integer,scaled,width,num_warps=1,num_stages=1)
        for phase in (0,1):
            if phase:
                q=q+.25;k=k.roll(1,0)
                qg.copy_(q);kg.copy_(k);sg.copy_(scales.flip(0));scales=scales.flip(0)
            expected_scale=q.float().abs().amax(1).clamp(min=1.e-8)/127
            packed=torch.round(q.float()/expected_scale[:,None]).clamp(-127,127).to(torch.int8)
            expected_integer=packed.long()@k.long()
            expected_float=expected_integer.float()*expected_scale[:,None]*scales[None,:]
            rng=torch.cuda.get_rng_state().clone();graph.replay();torch.cuda.synchronize()
            assert torch.equal(q8.cpu(),packed)
            assert torch.equal(qs.cpu(),expected_scale)
            assert torch.equal(integer.cpu().long(),expected_integer)
            actual=scaled.cpu();delta=(actual-expected_float).abs()
            assert torch.isfinite(actual).all() and bool((delta<=1.e-4+1.e-5*expected_float.abs()).all())
            assert torch.equal(torch.cuda.get_rng_state(),rng)
            row['phases'].append(dict(phase=phase,Q8_exact=True,Q_scale_exact=True,INT32_product_exact=True,
                                      float_max_abs=float(delta.max()),RNG_unchanged=True))
        result['cases'].append(row);save()
        del graph,qg,kg,sg,q8,qs,integer,scaled
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==h for name,h in result['frozen_sources'].items())
    result.update(complete=True,successful_shapes=sum(len(x['phases'])==2 for x in result['cases']))
except BaseException as error:
    result.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    result['ended_utc']=datetime.now(timezone.utc).isoformat();save()
    (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
