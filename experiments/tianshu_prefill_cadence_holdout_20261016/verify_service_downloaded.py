# SPDX-License-Identifier: Apache-2.0
"""Recompute the original-command bounded cadence screen and client stream intervals."""
import argparse
import json
from pathlib import Path
import re
import statistics
from audit_profile import audit_runtime


def read(path):
    return json.loads(path.read_text())


def metric_values(path):
    result={}
    for line in path.read_text().splitlines():
        m=re.match(r'^(vllm:[a-zA-Z0-9_]+)(?:\{[^}]*\})? ([0-9.eE+\-]+)$',line)
        if m:result[m[1]]=result.get(m[1],0.)+float(m[2])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    comparison=read(root/'comparison.json');assert comparison['complete']
    repo=Path(__file__).resolve().parents[2]
    import hashlib
    modes={};server_latency={}
    for mode in ('control','candidate'):
        run=root/mode;state=read(run/'result.json');assert state['phase']=='complete' and state['mode']==mode
        assert len(state['frozen_sources'])>=65
        assert '/usr/local/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py' in state['frozen_sources']
        profile=audit_runtime(run,state,official=True,online=False)
        assert profile['original_sampler']
        assert profile['strictTopP32Kgeo'] and profile['M2048_three_projection_tiles']
        assert profile['prefill_cadence']==(mode=='candidate') and profile['actual_bounded_cadence_and_held_request_exclusion_verified']
        for source,sha in state['frozen_sources'].items():
            if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
                assert hashlib.sha256((repo/Path(source).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
        rows=state['short_4k_rounds'];assert len(rows)==3
        all_seeds=[];server_rows=[]
        for i,row in enumerate(rows):
            responses=row['responses'];assert row['round']==i+1 and row['successful_requests']==len(responses)==128
            assert row['actual_queue_replenishment_verified']
            assert any(r['started_perf_counter']<max(x['finished_perf_counter'] for x in responses[:64]) for r in responses[64:])
            assert all(r['started_perf_counter']<r['finished_perf_counter'] and abs(r['finished_perf_counter']-r['started_perf_counter']-r['wall_seconds'])<1e-9 for r in responses)
            expected=[2026101700+1000*i+j for j in range(128)]
            assert [r['input_seed'] for r in responses]==expected
            all_seeds.extend(expected)
            for r in responses:
                assert r['api_seed'] is None
                assert r['temperature']==1.0
                assert (r['input_tokens'],r['output_tokens'])==(4096,1024)
                assert r['usage']['prompt_tokens']==4096 and r['usage']['completion_tokens']==1024 and r['usage']['total_tokens']==5120
                assert isinstance(r['text'],str) and r['text']
                assert 0<r['ttft_seconds']<=r['wall_seconds']<=row['duration_seconds']+.1
                offsets=r['text_event_offsets_seconds'];assert offsets and offsets[0]==r['ttft_seconds']
                assert all(0<a<=b<=r['wall_seconds'] for a,b in zip(offsets,offsets[1:]))
                gap=max((b-a for a,b in zip(offsets,offsets[1:])),default=0.)
                assert gap==r['max_text_event_gap_seconds']
                assert abs(r['mean_decode_token_interval_seconds']-(r['wall_seconds']-r['ttft_seconds'])/1023)<1e-12
            assert abs(row['total_tokens_per_second']-128*5120/row['duration_seconds'])<1e-9
            assert abs(row['output_tokens_per_second']-128*1024/row['duration_seconds'])<1e-9
            assert abs(row['mean_ttft_seconds']-statistics.mean(r['ttft_seconds'] for r in responses))<1e-12
            before_metrics=metric_values(run/f'round{i+1}_before_metrics.txt')
            after_metrics=metric_values(run/f'round{i+1}_after_metrics.txt')
            delta={k:after_metrics[k]-v for k,v in before_metrics.items() if k in after_metrics}
            assert delta['vllm:request_generation_tokens_count']==128
            assert delta['vllm:request_generation_tokens_sum']==128*1024
            assert delta['vllm:request_prompt_tokens_sum']==128*4096
            assert delta['vllm:time_to_first_token_seconds_count']==128
            assert delta['vllm:inter_token_latency_seconds_count']==128*1023
            ttft=1000*delta['vllm:time_to_first_token_seconds_sum']/128
            itl=1000*delta['vllm:inter_token_latency_seconds_sum']/(128*1023)
            assert ttft>0 and itl>0
            server_rows.append(dict(round=i+1,TTFT_ms=ttft,ITL_ms=itl,request_count=128,inter_token_count=128*1023))
        server_latency[mode]=server_rows
        assert len(set(all_seeds))==384
        assert abs(state['mean_total_tok_s']-statistics.mean(r['total_tokens_per_second'] for r in rows[1:]))<1e-9
        canaries=state['seeded_canaries'];assert len(canaries)==2
        assert canaries[0]['text']==canaries[1]['text']
        for c in canaries:assert c['api_seed']==777 and c['input_seed']==2026101699 and c['usage']['prompt_tokens']==128 and c['usage']['completion_tokens']==16
        concurrent=state['concurrent_canaries'];assert len(concurrent)==2
        for group in concurrent:
            assert len(group)==8
            for index,c in enumerate(group):
                assert c['input_seed']==2026101670+index and c['temperature']==0.0 and c['api_seed']==777
                assert c['usage']['prompt_tokens']==4096 and c['usage']['completion_tokens']==16
        gate=read(run/'canary_route_gate.json')
        assert gate['greedy_repeat_equal_count']==sum(a['text']==b['text'] for a,b in zip(*concurrent))
        metrics=list(run.glob('*metrics.txt'));assert len(metrics)==8
        for p in metrics:
            vals=re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$',p.read_text(),re.M)
            assert vals and all(float(v)==0 for v in vals)
        modes[mode]=state
    control,candidate=modes['control'],modes['candidate']
    assert control['frozen_sources']==candidate['frozen_sources']==comparison['frozen_sources']
    assert control['server_command']==candidate['server_command']
    for key,value in control['profile_environment'].items():
        other=candidate['profile_environment'][key]
        if key=='TIANSHU_PREFILL_CADENCE_MODE':assert (value,other)==('control','candidate')
        else:assert value.replace(control['output_root'],'<RUN>')==other.replace(candidate['output_root'],'<RUN>'),key
    assert control['seeded_canaries'][0]['text']==candidate['seeded_canaries'][0]['text']
    assert comparison['concurrent_greedy_cross_mode_equal_count']==sum(a['text']==b['text'] for ca,cb in zip(control['concurrent_canaries'],candidate['concurrent_canaries']) for a,b in zip(ca,cb))
    before,after=control['mean_total_tok_s'],candidate['mean_total_tok_s']
    assert comparison['control_total_tok_s']==before and comparison['candidate_total_tok_s']==after
    assert comparison['gain_percent']==100*(after/before-1)
    report=dict(complete=True,all768_generation_texts_usage_seeds_and_timing_checked=True,
        all_source_snapshots_and_current_repo_sources_checked=True,actual_original2048_256seq75warm_CV_KV16_S8_TopP_modes_verified=True,
        actual40_environment_one_cadence_switch_only_verified=True,actual_candidate_B64_strictTopP32Kgeo_and_three_M2048_configs_verified=True,
        all4_seeded_canaries_equal=True,all16_metrics_zero_preemptions=True,
        control_total_tok_s=before,candidate_total_tok_s=after,gain_percent=comparison['gain_percent'],
        official_benchmark=False,fresh_Math_not_run=True,RNG_unchanged=True,actual_cadence_route_bounds_and_exclusion_verified=True,all6_server_latency_token_and_request_counts_checked=True,server_latency_ms=server_latency)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
