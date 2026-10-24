# SPDX-License-Identifier: Apache-2.0
"""Preserve the failed N128 latency gate; ask a new full-workload question."""
from pathlib import Path
from candidate_profile import read,digest
BASE=Path(__file__).resolve().parent
SCREEN=BASE.parent/'tianshu_r5_BM8_fresh128_20261024'
PAIR=Path('/dev/shm/tianshu_r5_BM8_runs_20261024/fresh128_pair')

def check():
    gate=read(SCREEN/'SCREEN_GATE_32g.json')
    assert gate['complete'] and gate['model_quality_qualified'] and gate['all1024_performance_requests_succeeded']
    assert gate['paired_total_throughput_change_percent']>=1 and all(r['spread_percent']<1 for r in gate['arms'].values())
    assert gate['paired_TTFT_change_percent']>2 and not gate['meets_throughput_spread_and_TTFT_gate']
    assert (PAIR/'phase.txt').read_text().strip()=='BM8_r5_fresh128_complete_pending_review'
    states={arm:read(PAIR/arm/'result.json') for arm in ('control','candidate')}
    assert all(s['phase']=='complete' and s['evaluation_exit_code']==0 for s in states.values())
    assert states['control']['frozen_sources']==states['candidate']['frozen_sources']
    assert len(states['candidate']['frozen_sources'])==1023
    return dict(original_screen_qualified=False,original_failed_TTFT_gate_preserved=True,
        original_screen_gate_path=str(SCREEN/'SCREEN_GATE_32g.json'),original_screen_gate_sha256=digest(SCREEN/'SCREEN_GATE_32g.json'),
        N128_throughput_change_percent=gate['paired_total_throughput_change_percent'],N128_TTFT_change_percent=gate['paired_TTFT_change_percent'],
        research_question='Does the throughput/TTFT trade persist on the official N256 workload?',
        unchanged_decision_thresholds=dict(throughput_gain_at_least_percent=1,TTFT_regression_at_most_percent=2,hot_spread_below_percent=1),
        model_sources=states['candidate']['frozen_sources'],server_command=states['candidate']['server_command'])
