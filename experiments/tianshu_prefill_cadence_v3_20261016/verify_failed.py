# SPDX-License-Identifier: Apache-2.0
"""Audit completed control workload, retaining its failed postflight status."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import statistics
import audit_profile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve();read=lambda p:json.loads(p.read_text())
    assert read(root/'state.json')['phase']=='failed' and not (root/'candidate').exists()
    run=root/'control';state=read(run/'state.json')
    assert state['phase']=='failed' and state['error']=='AssertionError()'
    assert 'flush_sentinel' not in state and not (run/'after_metrics.txt').exists()
    assert not (root/'comparison.json').exists()
    assert len(state['frozen_sources'])==70
    repo=Path(__file__).resolve().parents[2]
    for name,sha in state['frozen_sources'].items():
        assert hashlib.sha256((run/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==sha
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
    # Last measured workload metrics exist, final post-diagnostic metrics do not.
    # Change only this audit's end-metric input; do not create a replacement file
    # or rewrite either the frozen auditor or the failed experiment state.
    source=inspect.getsource(audit_profile.audit_runtime)
    before='for name in ("before_metrics.txt", "after_metrics.txt"):'
    after='for name in ("before_metrics.txt", "round3_after_metrics.txt"):'
    assert source.count(before)==1
    namespace=dict(vars(audit_profile))
    exec(compile(source.replace(before,after),'<completed-workload-only end-metric audit>','exec'),namespace)
    runtime=namespace['audit_runtime'](run,state,official=True,online=False)
    assert runtime['complete'] and not runtime['prefill_cadence']
    assert len(state['short_4k_rounds'])==3
    timings=[];seeds=[]
    for i,row in enumerate(state['short_4k_rounds']):
        responses=row['responses'];assert row['round']==i+1 and row['successful_requests']==len(responses)==64
        expected=[2026101600+100*i+j for j in range(64)]
        assert [r['input_seed'] for r in responses]==expected;seeds.extend(expected)
        for r in responses:
            assert r['api_seed'] is None and r['temperature']==1.
            assert (r['input_tokens'],r['output_tokens'])==(4096,1024)
            assert r['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120)
            assert isinstance(r['text'],str) and r['text']
            assert 0<r['ttft_seconds']<=r['wall_seconds']<=row['duration_seconds']+.1
            offsets=r['text_event_offsets_seconds'];assert offsets and offsets[0]==r['ttft_seconds']
            assert all(0<a<=b<=r['wall_seconds'] for a,b in zip(offsets,offsets[1:]))
            assert r['max_text_event_gap_seconds']==max((b-a for a,b in zip(offsets,offsets[1:])),default=0.)
            assert abs(r['mean_decode_token_interval_seconds']-(r['wall_seconds']-r['ttft_seconds'])/1023)<1e-12
        assert row['total_tokens_per_second']==64*5120/row['duration_seconds']
        assert row['output_tokens_per_second']==64*1024/row['duration_seconds']
        assert row['mean_ttft_seconds']==statistics.mean(r['ttft_seconds'] for r in responses)
        assert not row['tuning_records']
        timings.append(dict(round=i+1,total_tok_s=row['total_tokens_per_second'],TTFT_ms=1000*row['mean_ttft_seconds']))
    assert len(set(seeds))==192
    metrics=list(run.glob('*metrics.txt'));assert len(metrics)==7
    for p in metrics:
        values=re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$',p.read_text(),re.M)
        assert values and all(float(v)==0 for v in values)
    report=dict(audit_complete=True,experiment_phase_remains_failed=True,complete_AB=False,candidate_not_run=True,
                all70_snapshots_and_owned_sources_checked=True,all192_control_responses_recomputed=True,
                end_metric_scope='round3_after_metrics, not missing post-diagnostic after_metrics',
                frozen_auditor_not_edited=True,all7_workload_metrics_zero_preemptions=True,
                runtime_workload_audit=runtime,timings=timings,
                historical_control_mean_tok_s=statistics.mean(r['total_tokens_per_second'] for r in state['short_4k_rounds'][1:]),
                final1token_SSE_not_serialized=True,new_formal_or_Math=False,adopted=False)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
