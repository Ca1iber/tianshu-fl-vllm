# SPDX-License-Identifier: Apache-2.0
"""New tile/grouping scan on M2048 BF16 model projections, no service binding."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import signal
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    lease = Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    remaining = (datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])-datetime.now(timezone.utc)).total_seconds()
    assert remaining > 2700
    signal.alarm(2700)
    hardware = subprocess.check_output(['ixsmi'], text=True)
    assert 'No running processes found' in hardware
    root.mkdir(parents=True)
    (root/'hardware_before.txt').write_text(hardware)
    import torch
    import vllm_fl  # noqa: F401
    import flag_gems
    from experimental_mm import mm
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    assert getattr(flag_gems,'current_work_registrar',None) is None
    configs = [dict(bm=bm,bn=bn,bk=bk,splits=1,warps=w,stages=s,group_m=g)
               for bm,bn,bk,w,s,g in (
                   (256,256,32,16,2,1),  # same arithmetic/tile as previous custom probe
                   (256,256,32,16,2,8),
                   (256,128,64,8,2,8),
                   (128,256,64,8,2,8),
                   (128,128,128,8,2,8),
                   (256,256,64,16,1,8),
                   (256,256,64,16,2,8),
                   (256,256,128,16,1,8),
                   (128,256,128,16,2,8),
                   (256,128,128,16,2,8),
                   (128,512,32,16,2,8),
                   (512,128,32,16,2,8))]
    paths=[Path(__file__).resolve(),Path(__file__).with_name('experimental_mm.py').resolve(),Path(inspect.getsourcefile(flag_gems.mm))]
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for i,p in enumerate(paths):
        target=root/'source_snapshot'/f'{i:02d}_{p.name}'
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    data=dict(complete=False,official_benchmark=False,adopted=False,cases=[],configs=configs,
              frozen_sources=frozen,scope='Synthetic M2048 custom Triton tiles; no model binding',
              no_flaggems_aten_registrar=True,tf32=False,bf16_reduced_precision_reduction=False,
              installed_sources_modified=False)

    def save():
        p=root/'result.tmp';p.write_text(json.dumps(data,indent=2)+'\n');p.replace(root/'result.json')

    def check(actual,reference):
        a,b=actual.float(),reference.float()
        delta=(a-b).abs()
        bad=delta>.02+.01*b.abs()
        return dict(passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),
                    failed_elements=int(bad.sum()),max_abs=float(delta.max()),atol=.02,rtol=.01)

    save()
    with torch.inference_mode():
        for name,n,k in [('qkv',2560,2048),('o',2048,2048),('gateup',12288,2048),('down',2048,6144)]:
            m=2048
            generator=torch.Generator(device='cuda').manual_seed(2026101430+n+k)
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16,generator=generator)
            weight=torch.randn(n,k,device='cuda',dtype=torch.bfloat16,generator=generator)
            rows=torch.linspace(0,m-1,8).long().cuda()
            cols=torch.linspace(0,n-1,32).long().cuda()
            for config in configs:
                item=dict(projection=name,M=m,N=n,K=k,config=config)
                graphs=[];outputs=[]
                try:
                    reference=x.float()@weight.float().T
                    cpu=x[rows].cpu().float()@weight[cols].cpu().float().T
                    output=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
                    scratch=(output,output)  # split1 never accesses the partial pointer
                    functions=[lambda:flag_gems.mm(x,weight.T),lambda:mm(x,weight,scratch,config)]
                    for function in functions:
                        for _ in range(3):actual=function()
                        assert check(actual,reference)['passed']
                        assert check(actual[rows[:,None],cols[None,:]].cpu(),cpu)['passed']
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(5):actual=function()
                        graph.replay();torch.cuda.synchronize()
                        assert check(actual,reference)['passed']
                        graphs.append(graph);outputs.append(actual)
                    x.neg_();weight.mul_(.5)
                    reference=x.float()@weight.float().T
                    cpu=x[rows].cpu().float()@weight[cols].cpu().float().T
                    for graph in graphs:graph.replay()
                    torch.cuda.synchronize()
                    fp32=[check(v,reference) for v in outputs]
                    cpus=[check(v[rows[:,None],cols[None,:]].cpu(),cpu) for v in outputs]
                    assert all(v['passed'] for v in fp32+cpus)
                    measurements=[[],[]]
                    for repeat in range(7):
                        for index in ((0,1) if repeat%2==0 else (1,0)):
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(3):graphs[index].replay()
                            end.record();end.synchronize()
                            measurements[index].append(start.elapsed_time(end)/15)
                    medians=list(map(statistics.median,measurements))
                    item.update(numerical_passed=True,changed_Graph_inputs_checked=True,
                                fp32_checks=fp32,cpu_checks=cpus,raw_ms=measurements,median_ms=medians,
                                time_change_percent=100*(medians[1]/medians[0]-1))
                    # Restore the exact BF16 values for the next config.
                    x.neg_();weight.mul_(2)
                except Exception as error:
                    item['error']=repr(error)
                    if any(v in str(error).lower() for v in ('illegal memory','device-side assert')):raise
                    # Fresh deterministic values even if a failure happened after mutation.
                    generator.manual_seed(2026101430+n+k)
                    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16,generator=generator)
                    weight=torch.randn(n,k,device='cuda',dtype=torch.bfloat16,generator=generator)
                data['cases'].append(item);save()
                print(json.dumps({k:v for k,v in item.items() if k not in ('raw_ms','fp32_checks','cpu_checks')}),flush=True)
                graphs=[];outputs=[];actual=None;reference=None;output=None;scratch=None
                torch.cuda.empty_cache()
    assert len(data['cases'])==48
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in frozen.items())
    data.update(complete=True,numerically_passing_cases=sum(v.get('numerical_passed',False) for v in data['cases']))
    save()


if __name__=='__main__':
    main()
