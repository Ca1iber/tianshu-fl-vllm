# SPDX-License-Identifier: Apache-2.0
"""Audit a control-only greedy-repeat gate failure, not an optimization result."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve();read=lambda p:json.loads(p.read_text())
    assert read(root/'state.json')['phase']=='failed' and not (root/'candidate').exists()
    run=root/'control';state=read(run/'state.json')
    assert state['phase']=='failed' and state['error']=='AssertionError()'
    assert not state['short_4k_rounds'] and not (root/'comparison.json').exists()
    repo=Path(__file__).resolve().parents[2]
    assert len(state['frozen_sources'])==69
    for name,sha in state['frozen_sources'].items():
        assert hashlib.sha256((run/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==sha
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
    records=[read(p) for p in (run/'prefill_cadence_records').glob('runtime_*.json')]
    used=[r for r in records if r['calls']>0]
    assert len(used)==1 and used[0]['supported']>0 and used[0]['deferred_steps']==0
    assert used[0]['configuration']['configured_use_v2'] is True
    assert used[0]['configuration']['worker_cls']=='vllm_fl.worker.worker.WorkerFL'
    groups=state['concurrent_canaries'];assert len(groups)==2 and all(len(g)==8 for g in groups)
    for group in groups:
        for i,r in enumerate(group):
            assert r['input_seed']==2026101670+i and r['api_seed']==777 and r['temperature']==0.
            assert r['usage']==dict(prompt_tokens=4096,completion_tokens=16,total_tokens=4112)
            assert isinstance(r['text'],str) and r['text']
            assert 0<r['ttft_seconds']<r['wall_seconds']
    mismatches=[i for i,(a,b) in enumerate(zip(*groups)) if a['text']!=b['text']]
    assert mismatches==[0,2,4,5]
    c=state['seeded_canaries'];assert len(c)==2 and c[0]['text']==c[1]['text']
    reason=read(root/'stop_reason.json');assert reason['original_state_not_rewritten'] and reason['greedy_control_repeat_mismatches']==4
    assert 'No running processes found' in (run/'hardware_after.txt').read_text()
    report=dict(complete=True,failed_status_not_rewritten=True,all69_snapshots_and_owned_sources_verified=True,
                all16_greedy_concurrent_responses_checked=True,control_repeat_mismatch_indices=mismatches,
                actual_configured_V2_flag_and_WorkerFL_observed=True,
                candidate_not_run=True,throughput_rounds=0,quality_and_cause_not_inferred=True,adopted=False)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
