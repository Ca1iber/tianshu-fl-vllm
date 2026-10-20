# SPDX-License-Identifier: Apache-2.0
"""Production FL block + CpuGpuBuffer versus fused integer metadata; no model."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from types import SimpleNamespace as NS


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>180
    signal.alarm(min(1800,remaining))
    hardware=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_before.txt').write_text(hardware)
    assert 'No running processes found' in hardware
    import numpy as np
    import torch
    import vllm_fl
    import flag_gems
    from vllm.v1.utils import CpuGpuBuffer
    from source import BASE,NATIVE,baseline_function,eligible,render
    from kernel import launch
    torch.set_num_threads(1)
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['CPU_only'] and gate['CPU_state_comparisons']==384
    for path,sha in gate['source_sha256'].items():assert digest(path)==sha,path
    paths=[*BASE.glob('*.py'),BASE/'CPU_GATE_32g.json',NATIVE,Path(inspect.getsourcefile(CpuGpuBuffer)),
           Path(flag_gems.__file__),Path('/usr/local/lib/python3.12/site-packages/triton/backends/iluvatar/compiler.py')]
    for name in ('index','add','fill_','copy'):
        fn=getattr(flag_gems,name,None)
        if fn is not None:
            try: path=inspect.getsourcefile(fn)
            except TypeError: path=inspect.getsourcefile(type(fn))
            if path:paths.append(Path(path))
    frozen={str(path.resolve()):digest(path) for path in paths}
    for name,sha in frozen.items():
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    # Same default all-op FlagGems registration as WorkerFL; explicit component scope.
    flag_gems.enable(record=True,once=True,path=str(root/'flag_gems_oplist.txt'))
    data=dict(complete=False,official_benchmark=False,adopted=False,runner_pid=os.getpid(),cases=[],frozen_sources=frozen,
              production_FL_block_source=str(NATIVE),production_FL_block_sha256=digest(NATIVE),
              actual_CpuGpuBuffer=True,FlagGems_all_ops_enabled=True,CPU_reference_after_device_to_host=True,
              no_model_service_or_Math=True,source_provenance=render()[1])
    baseline=baseline_function()
    def save():(root/'result.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    save()
    def owner(device,capacity,phase,n):
        comp=(torch.arange(capacity,dtype=torch.int32)*13+phase*4091)%131072
        def buffer(length,dtype):
            b=CpuGpuBuffer(length,dtype=dtype,device=torch.device(device),pin_memory=(device=='cuda'))
            b.cpu.copy_(torch.arange(length,dtype=dtype)+701);b.gpu.copy_(b.cpu+1000);return b
        result=NS(num_computed_tokens=comp.to(device),req_indices=buffer(4096,torch.int64),
            query_pos=buffer(4096,torch.int64),num_scheduled_tokens=buffer(capacity,torch.int32),
            positions=(torch.arange(4096,dtype=torch.int64)-100).to(device),seq_lens=(torch.arange(capacity,dtype=torch.int32)+100).to(device),
            uses_mrope=False,uses_xdrope_dim=0,speculative_config=None,enable_prompt_embeds=False,
            mamba_prev_last_scheduled_idx=None,use_async_spec_decode=False)
        result.query_pos.np[:n]=0
        return result
    def candidate(o,indices,counts,n,total):
        if not eligible(o,counts,n,total):return baseline(o,indices,counts,n,total)
        o.req_indices.np[:total]=indices;o.num_scheduled_tokens.np[:n]=counts
        return launch(o,n)
    def arrays(o):
        result={}
        for name in ('req_indices','query_pos','num_scheduled_tokens'):
            pair=getattr(o,name);result[name+'_CPU']=pair.cpu.clone();result[name+'_GPU']=pair.gpu.cpu()
        for name in ('positions','seq_lens','num_computed_tokens'):result[name]=getattr(o,name).cpu()
        return result
    def verify(o,expected):
        actual=arrays(o);assert actual.keys()==expected.keys()
        for name,value in actual.items():assert torch.equal(value,expected[name]),name
        return {name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in actual.items()}
    def event_time(graph,count=100):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):graph.replay()
        end.record();end.synchronize();return start.elapsed_time(end)/count
    def host_time(fn,count=25):
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(count):fn()
        torch.cuda.synchronize();return 1000*(time.perf_counter()-start)/count
    try:
        with torch.inference_mode():
            rng=torch.cuda.get_rng_state().clone()
            for capacity in (64,256):
                for n in (1,8,32,40,41,63,64):
                    indices=np.arange(n,dtype=np.int64);counts=np.ones(n,dtype=np.int32)
                    a,b=owner('cuda',capacity,0,n),owner('cuda',capacity,0,n)
                    assert eligible(a,counts,n,n) and eligible(b,counts,n,n)
                    original=b.num_computed_tokens.cpu().clone()
                    baseline(a,indices,counts,n,n);compiled=candidate(b,indices,counts,n,n);torch.cuda.synchronize()
                    ref=owner('cpu',capacity,0,n);baseline(ref,indices,counts,n,n);expected=arrays(ref)
                    proof=[dict(phase=0,control=verify(a,expected),candidate=verify(b,expected))]
                    assert torch.equal(b.num_computed_tokens.cpu(),original)
                    meta=dict(num_warps=compiled.metadata.num_warps,num_stages=compiled.metadata.num_stages,
                              shared=compiled.metadata.shared,n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),
                              grid=[(capacity+255)//256],BLOCK=256,N=n,CAPACITY=capacity,source_sha256=digest(BASE/'kernel.py'))
                    assert meta['num_warps']==4 and meta['num_stages']==1
                    ga,gb=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
                    with torch.cuda.graph(ga):baseline(a,indices,counts,n,n)
                    with torch.cuda.graph(gb):candidate(b,indices,counts,n,n)
                    for phase in (1,31):
                        updated=((torch.arange(capacity,dtype=torch.int32)*13+phase*4091)%131072).flip(0)
                        a.num_computed_tokens.copy_(updated);b.num_computed_tokens.copy_(updated)
                        ref=owner('cpu',capacity,phase,n);ref.num_computed_tokens.copy_(updated)
                        baseline(ref,indices,counts,n,n);expected=arrays(ref)
                        ga.replay();gb.replay();torch.cuda.synchronize()
                        proof.append(dict(phase=phase,control=verify(a,expected),candidate=verify(b,expected)))
                    graph_ms={'control':[],'candidate':[]};wall_ms={'control':[],'candidate':[]}
                    work={'control':lambda:baseline(a,indices,counts,n,n),'candidate':lambda:candidate(b,indices,counts,n,n)}
                    for repeat in range(7):
                        order=('control','candidate') if repeat%2==0 else ('candidate','control')
                        for mode in order:
                            graph_ms[mode].append(event_time(ga if mode=='control' else gb))
                            wall_ms[mode].append(host_time(work[mode]))
                    assert torch.equal(torch.cuda.get_rng_state(),rng)
                    for o in (a,b):verify(o,expected)
                    row=dict(capacity=capacity,batch=n,phases=proof,compiled_metadata=meta,same_Graph_changed_inputs_verified=True,
                        full_buffers_and_tails_bitwise=True,RNG_unchanged=True,
                        graph_raw_ms=graph_ms,wall_raw_ms=wall_ms,
                        graph_median_ms={k:statistics.median(v) for k,v in graph_ms.items()},
                        wall_median_ms={k:statistics.median(v) for k,v in wall_ms.items()})
                    data['cases'].append(row);save()
                    print(json.dumps({k:row[k] for k in ('capacity','batch','graph_median_ms','wall_median_ms')}),flush=True)
            for path,sha in frozen.items():assert digest(path)==sha,path
            data.update(complete=True,full_buffer_checks=14*3*2,CPU_model_state_cases=384)
    except BaseException as error:
        data['error']=repr(error);save();raise
    finally:
        save()
    return 0


if __name__=='__main__':raise SystemExit(main())
