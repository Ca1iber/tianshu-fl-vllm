# SPDX-License-Identifier: Apache-2.0
"""CPU bitwise metadata reference, protected tails, alternating GPU graphs."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    assert not (root/'result.json').exists()
    lease=Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    remaining=int((datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])-datetime.now(timezone.utc)).total_seconds())
    assert remaining>0;signal.alarm(min(1800,remaining))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    (root/'hardware_before.txt').write_text(hardware)
    import torch
    import vllm_fl
    import flag_gems
    from experimental_metadata import fill
    torch.set_num_threads(8)
    result={'complete':False,'official_benchmark':False,'adoption_allowed':False,
            'installed_files_modified':False,'cases':[],
            'source_sha256':hashlib.sha256(Path(__file__).with_name('experimental_metadata.py').read_bytes()).hexdigest()}
    def save():(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    cases=((1,1),(1,565),(1,2048),(1,8192),(8,8),(8,64),(8,2048),
           (16,16),(16,2048),(64,64),(64,565),(64,2048),(256,256),(256,1537),(256,2048))
    try:
        with torch.inference_mode():
            for index,(num_reqs,num_tokens) in enumerate(cases):
                max_reqs=256 if index%2==0 else 1024
                generator=torch.Generator().manual_seed(2026101270+index)
                counts_cpu=torch.randint(0,131000,(max_reqs,),dtype=torch.int32,generator=generator)
                counts_cpu[0]=2147483647 # Verify casting occurs before int64 position addition.
                ids_cpu=torch.arange(num_tokens,dtype=torch.int64)%num_reqs
                query_cpu=torch.arange(num_tokens,dtype=torch.int64)//num_reqs
                scheduled_cpu=torch.bincount(ids_cpu,minlength=max_reqs).to(torch.int32)
                expected_positions=counts_cpu[ids_cpu].to(torch.int64)+query_cpu
                expected_lens=torch.zeros(max_reqs,dtype=torch.int32)
                expected_lens[:num_reqs]=counts_cpu[:num_reqs]+scheduled_cpu[:num_reqs]
                counts=counts_cpu.cuda();ids=ids_cpu.cuda();query=query_cpu.cuda();scheduled=scheduled_cpu.cuda()
                before_pos=torch.full((num_tokens+13,),-777,device='cuda',dtype=torch.int64)
                after_pos=before_pos.clone();before_lens=torch.full((max_reqs+13,),-888,device='cuda',dtype=torch.int32)
                after_lens=before_lens.clone()
                def baseline():
                    before_pos[:num_tokens].copy_(counts[ids].to(torch.int64)+query)
                    before_lens[:num_reqs].copy_(counts[:num_reqs]+scheduled[:num_reqs])
                    before_lens[num_reqs:max_reqs].fill_(0)
                with flag_gems.use_gems():baseline()
                fill(counts,ids,query,scheduled,after_pos,after_lens,num_tokens,num_reqs)
                checks={'candidate_positions_cpu_equal':torch.equal(after_pos[:num_tokens].cpu(),expected_positions),
                        'candidate_seq_lens_cpu_equal':torch.equal(after_lens[:max_reqs].cpu(),expected_lens),
                        'baseline_positions_cpu_equal':torch.equal(before_pos[:num_tokens].cpu(),expected_positions),
                        'baseline_seq_lens_cpu_equal':torch.equal(before_lens[:max_reqs].cpu(),expected_lens),
                        'position_tail_preserved':bool((after_pos[num_tokens:]==-777).all()),
                        'seq_lens_tail_preserved':bool((after_lens[max_reqs:]==-888).all())}
                item={'num_reqs':num_reqs,'num_tokens':num_tokens,'max_reqs':max_reqs,'checks':checks,'passed':all(checks.values())}
                if item['passed']:
                    graphs=[]
                    for selected in (0,1):
                        with flag_gems.use_gems():
                            for _ in range(3):
                                if selected:fill(counts,ids,query,scheduled,after_pos,after_lens,num_tokens,num_reqs)
                                else:baseline()
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                for _ in range(30):
                                    if selected:fill(counts,ids,query,scheduled,after_pos,after_lens,num_tokens,num_reqs)
                                    else:baseline()
                        graphs.append(graph)
                    raw=[[],[]]
                    for repetition in range(7):
                        for selected in ((0,1) if repetition%2==0 else (1,0)):
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record();graphs[selected].replay();end.record();end.synchronize()
                            raw[selected].append(start.elapsed_time(end)/30)
                    before,after=map(statistics.median,raw)
                    item.update(baseline_ms=before,candidate_ms=after,change_percent=100*(after/before-1),raw_ms=raw)
                result['cases'].append(item);print(json.dumps(item),flush=True);save()
            result.update(complete=True,cases_passed=sum(x['passed'] for x in result['cases']),cases_total=len(result['cases']))
    finally:save()


if __name__=='__main__':main()
