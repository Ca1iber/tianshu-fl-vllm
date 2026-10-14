# SPDX-License-Identifier: Apache-2.0
"""Diagnostic API comparison only: FlagGems.mm and torch.mm on model shapes."""
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
    assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
    deadline = datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    assert (deadline-datetime.now(timezone.utc)).total_seconds()>1800
    signal.alarm(1800)
    root.mkdir(parents=True)
    import torch
    import vllm_fl  # noqa: F401
    import flag_gems
    assert getattr(flag_gems, 'current_work_registrar', None) is None
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(2026101417)
    sources=[Path(__file__).resolve(),Path(inspect.getsourcefile(flag_gems.mm)),Path(flag_gems.__file__)]
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for i,p in enumerate(sources):
        target=root/'source_snapshot'/f'{i:02d}_{p.name}'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    (root/'aten_mm_dispatch.txt').write_text(torch._C._dispatch_dump_table('aten::mm'))
    (root/'hardware_before.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
    cfg=json.loads(Path('/workspace/MiniCPM5-2B/config.json').read_text())
    assert cfg['hidden_size']==2048 and cfg['intermediate_size']==6144 and cfg['vocab_size']==130560
    data=dict(complete=False,official_benchmark=False,adopted=False,cases=[],frozen_sources=frozen,
              torch_version=torch.__version__,flaggems_version=getattr(flag_gems,'__version__','unknown'),
              comparison=['flag_gems.mm','torch.mm'],tf32=False,bf16_reduced_precision_reduction=False,
              no_flaggems_aten_registrar=True,installed_sources_modified=False,
              scope='Synthetic operator API diagnostic; no model binding, rules/adoption, or service claim')

    def save():
        p=root/'result.tmp'
        p.write_text(json.dumps(data,indent=2)+'\n')
        p.replace(root/'result.json')

    def check(a,b):
        a,b=a.float(),b.float()
        diff=(a-b).abs()
        bad=diff>.02+.01*b.abs()
        return dict(passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),
                    max_abs=diff.max().item(),failed_elements=bad.sum().item(),atol=.02,rtol=.01)

    projections=[('qkv',2560,2048),('o',2048,2048),('gateup',12288,2048),('down',2048,6144),('head',130560,2048)]
    shapes=[(name,m,n,k) for m in (8,16,32,64) for name,n,k in projections]
    shapes += [(name,2048,n,k) for name,n,k in projections if name!='head']
    save()
    with torch.inference_mode():
        for name,m,n,k in shapes:
            item=dict(projection=name,M=m,N=n,K=k)
            try:
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                weight=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
                rows=torch.linspace(0,m-1,min(m,8)).long()
                cols=torch.linspace(0,n-1,32).long()
                functions=[lambda:flag_gems.mm(x,weight.T),lambda:torch.mm(x,weight.T)]
                reference=x.float()@weight.float().T
                cpu=x[rows.cuda()].cpu().float()@weight[cols.cuda()].cpu().float().T
                graphs=[]
                outputs=[]
                for function in functions:
                    for _ in range(3):actual=function()
                    assert check(actual,reference)['passed']
                    assert check(actual[rows.cuda()[:,None],cols.cuda()[None,:]].cpu(),cpu)['passed']
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(5):actual=function()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert check(actual,reference)['passed']
                    graphs.append(graph)
                    outputs.append(actual)
                # Same captured addresses, new BF16 values, independent FP32/CPU references.
                x.neg_()
                weight.mul_(.5)
                reference=x.float()@weight.float().T
                cpu=x[rows.cuda()].cpu().float()@weight[cols.cuda()].cpu().float().T
                for graph in graphs:graph.replay()
                torch.cuda.synchronize()
                checks=[check(o,reference) for o in outputs]
                cpu_checks=[check(o[rows.cuda()[:,None],cols.cuda()[None,:]].cpu(),cpu) for o in outputs]
                assert all(x['passed'] for x in checks+cpu_checks)
                measurements=[[],[]]
                for repeat in range(7):
                    for index in ((0,1) if repeat%2==0 else (1,0)):
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(3):graphs[index].replay()
                        end.record()
                        end.synchronize()
                        measurements[index].append(start.elapsed_time(end)/15)
                medians=list(map(statistics.median,measurements))
                item.update(numerical_passed=True,changed_Graph_inputs_checked=True,fp32_checks=checks,cpu_checks=cpu_checks,
                            raw_ms=measurements,median_ms=medians,time_change_percent=100*(medians[1]/medians[0]-1))
            except Exception as error:
                item['error']=repr(error)
                if any(x in str(error).lower() for x in ('illegal memory','device-side assert')):
                    raise
            data['cases'].append(item)
            save()
            print(json.dumps({k:v for k,v in item.items() if k not in ('raw_ms','cpu_checks','fp32_checks')}),flush=True)
            # Captured graphs own their allocations until released.
            graphs=[]
            outputs=[]
            actual=None
            reference=None
            x=None
            weight=None
            torch.cuda.empty_cache()
    assert len(data['cases'])==24
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in frozen.items())
    data.update(complete=True,all_numerical_checks_passed=all(c.get('numerical_passed',False) for c in data['cases']))
    save()


if __name__=='__main__':
    main()
