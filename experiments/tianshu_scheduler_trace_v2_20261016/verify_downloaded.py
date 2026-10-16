# SPDX-License-Identifier: Apache-2.0
"""Recompute diagnostic histograms without treating profiling as a score."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from audit_profile import audit_runtime


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve();state=read(root/'result.json')
    assert state['phase']=='complete' and state['diagnostic_only'] and state['throughput_not_comparable_due_to_observers']
    runtime=audit_runtime(root,state,official=True,online=False)
    assert runtime['diagnostic_observer_only']
    repo=Path(__file__).resolve().parents[2]
    for source,sha in state['frozen_sources'].items():
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
            assert hashlib.sha256((repo/Path(source).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
    rows=state['short_4k_rounds'];assert len(rows)==2
    for i,row in enumerate(rows):
        responses=row['responses'];assert len(responses)==row['successful_requests']==64
        assert [r['input_seed'] for r in responses]==[2026101600+100*i+j for j in range(64)]
        for r in responses:
            assert r['api_seed'] is None and r['text'] and r['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120)
        assert abs(row['total_tokens_per_second']-64*5120/row['duration_seconds'])<1e-9
    assert len(state['seeded_canaries'])==2 and state['seeded_canaries'][0]['text']==state['seeded_canaries'][1]['text']
    assert state['flush_sentinel']['usage']==dict(prompt_tokens=16,completion_tokens=1,total_tokens=17)
    directory=root/'scheduler_trace_records'
    steps=[];summaries=[read(p) for p in directory.glob('summary_*.json')]
    for path in directory.glob('steps_*.jsonl'):
        current=[json.loads(line) for line in path.read_text().splitlines()]
        assert [s['step'] for s in current]==list(range(1,len(current)+1))
        steps.extend(current)
    assert steps and len([s for s in summaries if s['steps']>0])==1
    summary=next(s for s in summaries if s['steps']>0)
    assert summary['steps']==len(steps) and summary['observer_only'] and not summary['policy_changed']
    assert summary['timings_include_nested_observer_overhead']
    for source,sha in summary['observed_sources'].items():
        assert state['frozen_sources'][source]==sha
    categories=Counter();batches=Counter();alloc=Counter();single=Counter();throttled=0
    for s in steps:
        throttled+=int(s['throttle_prefills'])
        assert s['wall_ns']>0 and s['prefill_tokens']+s['decode_tokens']==s['scheduled_tokens']
        assert 0<=s['scheduled_reqs']<=256 and s['scheduled_tokens']<=2048
        assert s['preempted']==0
        assert s['before']['free_blocks']>=0 and s['after']['free_blocks']>=0
        category='mixed' if s['prefill_tokens'] and s['decode_tokens'] else 'prefill' if s['prefill_tokens'] else 'decode' if s['decode_tokens'] else 'empty'
        categories[category]+=1;batches[str(s['scheduled_reqs'])]+=1
        alloc.update(s['calls'].get('allocate_slots',{}));single.update(s['calls'].get('single_cache_blocks',{}))
    decode=[s for s in steps if s['decode_tokens'] and not s['prefill_tokens']]
    full=[s for s in decode if s['scheduled_reqs']==64]
    assert full and alloc['calls']>0 and single['calls']>0
    def timing(values):
        return dict(steps=len(values),median_schedule_ms=statistics.median(s['wall_ns'] for s in values)/1e6,
            median_allocate_ms=statistics.median(s['calls'].get('allocate_slots',{}).get('wall_ns',0) for s in values)/1e6)
    report=dict(complete=True,diagnostic_only=True,official_benchmark=False,adopted=False,
        all128_responses_and_frozen_sources_checked=True,all_trace_rows_checked=True,
        observed_source_count=len(summary['observed_sources']),trace_steps=len(steps),throttled_steps=throttled,categories=dict(categories),batch_histogram=dict(batches),
        allocation_totals=dict(alloc),single_cache_totals=dict(single),
        allocate_empty_percent=100*alloc['empty']/alloc['calls'],
        single_cache_early_percent=100*single['already_full']/single['calls'],
        outside_schedule_counters=summary['outside_schedule'],pure_decode=timing(decode),B64_pure_decode=timing(full),
        min_free_KV_blocks=min(s['after']['free_blocks'] for s in steps),max_waiting=max(s['before']['waiting'] for s in steps),
        nested_timings_must_not_be_added=True,unprofiled_CPU_or_GPU_cost_not_measured=True)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
