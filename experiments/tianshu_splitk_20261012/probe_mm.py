# SPDX-License-Identifier: Apache-2.0
"""Lease-bounded decode projection scan, independent CPU samples and graphs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess
import inspect
import importlib.metadata

def error(value, reference):
    delta = (value.float() - reference.float()).abs()
    failed = delta > (.02 + .01 * reference.float().abs())
    return {'passed': not bool(failed.any()), 'max_abs': float(delta.max()),
            'failed_elements': int(failed.sum()), 'atol': .02, 'rtol': .01}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'result.json').exists(), 'Refusing to overwrite prior scan'
    lease = Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    remaining = int((datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc']) - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 0
    signal.alarm(min(1800, remaining))
    hardware = subprocess.check_output(['ixsmi'], text=True)
    assert 'No running processes found' in hardware, 'GPU jobs must be serial'
    (root / 'hardware_before.txt').write_text(hardware)
    # Check ownership before compatibility imports can initialize our context.
    import torch
    import vllm_fl  # same installed Iluvatar compatibility initialization
    import flag_gems
    from experimental_mm import prepare, mm
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    configs = [dict(bm=bm, bn=bn, bk=64, splits=splits, warps=warps, stages=2)
               for bm, bn, warps in ((16,64,4),(32,64,4),(64,64,4),(32,128,4),(64,128,8))
               for splits in (1,2,4)]
    if args.smoke:
        configs=[c for c in configs if c['bm']==64 and c['bn']==64]
    result = {'complete': False, 'official_benchmark': False, 'adoption_allowed': False,
              'installed_files_modified': False, 'cases': [],
              'source_sha256': hashlib.sha256(Path(__file__).with_name('experimental_mm.py').read_bytes()).hexdigest()}
    actual=Path(inspect.getsourcefile(flag_gems.mm))
    result['baseline_source']={'path':str(actual),'sha256':hashlib.sha256(actual.read_bytes()).hexdigest()}
    result['packages']={name:importlib.metadata.version(name) for name in ('torch','flag-gems','vllm')}

    def save():
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    try:
        with torch.inference_mode():
            for m in ((64,) if args.smoke else (64, 8)):
                shapes=(('qkv',2560,2048),) if args.smoke else (('qkv',2560,2048),('o_proj',2048,2048),
                                                                             ('gate_up',12288,2048),('down',2048,6144))
                for name,n,k in shapes:
                    gen = torch.Generator(device='cuda').manual_seed(2026101250+m+n)
                    a = torch.randn((m,k),device='cuda',dtype=torch.bfloat16,generator=gen)
                    weight = torch.randn((n,k),device='cuda',dtype=torch.bfloat16,generator=gen)
                    reference = a.float() @ weight.float().T
                    rows = torch.linspace(0,m-1,min(m,8),device='cuda').long()
                    columns = torch.linspace(0,n-1,32,device='cuda').long()
                    cpu = a[rows].float().cpu() @ weight[columns].float().cpu().T
                    baseline = flag_gems.mm(a,weight.T)
                    basecheck = error(baseline.float(),reference)
                    cpu_base = error(baseline[rows[:,None],columns[None,:]].cpu(),cpu)
                    assert basecheck['passed'] and cpu_base['passed'], (basecheck,cpu_base)
                    for config in configs:
                        item = {'name':name,'m':m,'n':n,'k':k,'config':config}
                        try:
                            scratch = prepare(a,weight,config)
                            candidate = mm(a,weight,scratch,config)
                            item['vs_fp32'] = error(candidate,reference)
                            item['vs_cpu'] = error(candidate[rows[:,None],columns[None,:]].cpu(),cpu)
                            item['vs_native'] = error(candidate,baseline)
                            if item['vs_fp32']['passed'] and item['vs_cpu']['passed']:
                                for _ in range(3):
                                    flag_gems.mm(a,weight.T)
                                    mm(a,weight,scratch,config)
                                torch.cuda.synchronize()
                                graphs=[]
                                for function in (lambda:flag_gems.mm(a,weight.T),lambda:mm(a,weight,scratch,config)):
                                    graph=torch.cuda.CUDAGraph()
                                    with torch.cuda.graph(graph):
                                        for _ in range(20):function()
                                    graphs.append(graph)
                                raw=[[],[]]
                                for repetition in range(7):
                                    for selected in ((0,1) if repetition%2==0 else (1,0)):
                                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                        start.record();graphs[selected].replay();end.record();end.synchronize()
                                        raw[selected].append(start.elapsed_time(end)/20)
                                before,after=map(statistics.median,raw)
                                item.update(baseline_ms=before,candidate_ms=after,change_percent=100*(after/before-1),raw_ms=raw)
                        except Exception as exc:
                            item['error']=str(exc)
                            if any(s in str(exc).lower() for s in ('illegal memory','device-side assert')):raise
                        result['cases'].append(item)
                        print(json.dumps(item),flush=True)
                        save()
            result['complete']=True
    finally:
        save()


if __name__=='__main__':
    main()
