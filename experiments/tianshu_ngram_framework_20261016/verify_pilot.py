# SPDX-License-Identifier: Apache-2.0
"""Recompute the candidate-only pilot against its recent control reference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
from audit_pilot import audit_runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();control=args.control.resolve()
    def read(path):return json.loads(path.read_text())
    def metric(path):
        result={}
        for line in path.read_text().splitlines():
            m=re.match(r'^(vllm:[a-zA-Z0-9_]+)(?:\{[^}]*\})? ([0-9.eE+\-]+)$',line)
            if m:result[m[1]]=result.get(m[1],0.)+float(m[2])
        return result
    state=read(root/'result.json');reference=read(control/'result.json')
    assert state['phase']==reference['phase']=='complete'
    runtime=audit_runtime(root,state,online=False)
    assert state['reference_control_root']==reference['output_root'] and state['server_command'][:-2]==reference['server_command']
    overlap=set(reference['frozen_sources'])&set(state['frozen_sources']);assert len(overlap)>=58
    assert all(reference['frozen_sources'][name]==state['frozen_sources'][name] for name in overlap)
    repo=Path(__file__).resolve().parents[2]
    for name,h in state['frozen_sources'].items():
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==h
    rows=state['short_4k_rounds'];assert len(rows)==3;latency=[];allseeds=[]
    for i,row in enumerate(rows):
        responses=row['responses'];assert row['successful_requests']==len(responses)==64 and row['round']==i+1
        seeds=[2026102800+100*i+j for j in range(64)]
        assert [r['input_seed'] for r in responses]==seeds==[r['input_seed'] for r in reference['short_4k_rounds'][i]['responses']]
        allseeds.extend(seeds)
        for r in responses:
            assert r['api_seed'] is None and r['input_tokens']==4096 and r['output_tokens']==1024
            assert r['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120) or (
                r['usage']['prompt_tokens']==4096 and r['usage']['completion_tokens']==1024 and r['usage']['total_tokens']==5120)
            assert r['text'] and 0<r['ttft_seconds']<=r['wall_seconds']<=row['duration_seconds']+.1
        assert abs(row['total_tokens_per_second']-64*5120/row['duration_seconds'])<1e-9
        assert abs(row['output_tokens_per_second']-64*1024/row['duration_seconds'])<1e-9
        assert abs(row['mean_ttft_seconds']-statistics.mean(r['ttft_seconds'] for r in responses))<1e-12
        before,after=metric(root/f'round{i+1}_before_metrics.txt'),metric(root/f'round{i+1}_after_metrics.txt')
        delta={k:after[k]-v for k,v in before.items() if k in after}
        assert delta['vllm:request_generation_tokens_count']==64 and delta['vllm:request_generation_tokens_sum']==65536
        assert delta['vllm:request_prompt_tokens_sum']==262144 and delta['vllm:time_to_first_token_seconds_count']==64
        assert delta['vllm:inter_token_latency_seconds_count']==64*1023
        latency.append(dict(round=i+1,TTFT_ms=1000*delta['vllm:time_to_first_token_seconds_sum']/64,
            ITL_ms=1000*delta['vllm:inter_token_latency_seconds_sum']/(64*1023),
            speculative_counters={k:v for k,v in delta.items() if k.startswith('vllm:spec_decode_')}))
    assert len(set(allseeds))==192
    canaries=state['seeded_canaries'];assert len(canaries)==2 and canaries[0]['text']==canaries[1]['text']
    assert all(r['api_seed']==777 and r['input_seed']==2026101699 and r['usage']['completion_tokens']==16 for r in canaries)
    mean=statistics.mean(r['total_tokens_per_second'] for r in rows[1:]);assert mean==state['mean_total_tok_s']
    result=dict(complete=True,all192_responses_and_source_records_verified=True,runtime=runtime,
        control_reference_tok_s=reference['mean_total_tok_s'],candidate_tok_s=mean,
        gain_percent=100*(mean/reference['mean_total_tok_s']-1),server_latency=latency,
        separate_fresh_interleaved_AB=False,diagnostic_candidate_only_pilot=True,
        fresh_Math_not_run=True,official_benchmark=False,adopted=False)
    (root/'local_download_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
