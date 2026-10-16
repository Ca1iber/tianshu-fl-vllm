# SPDX-License-Identifier: Apache-2.0
"""Audit the guard-mismatch interruption without treating it as an A/B."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root.resolve();read=lambda path:json.loads(path.read_text())
    assert read(root/'state.json')['phase']=='failed'
    run=root/'control';state=read(run/'state.json')
    assert state['phase']=='stopped' and not (root/'candidate').exists()
    assert len(state['frozen_sources'])==69
    repo=Path(__file__).resolve().parents[2]
    for name,sha in state['frozen_sources'].items():
        assert hashlib.sha256((run/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==sha
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
    records=[read(p) for p in (run/'prefill_cadence_records').glob('runtime_*.json')]
    assert records and all(r['supported']==r['deferred_steps']==r['held_requests']==0 for r in records)
    rows=state['short_4k_rounds'];assert len(rows)==1;row=rows[0]
    assert row['round']==1 and row['successful_requests']==64 and len(row['responses'])==64
    assert [r['input_seed'] for r in row['responses']]==list(range(2026101600,2026101664))
    for r in row['responses']:
        assert r['api_seed'] is None and r['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120)
        assert isinstance(r['text'],str) and r['text']
        assert 0<r['ttft_seconds']<r['wall_seconds']<=row['duration_seconds']+.1
        offsets=r['text_event_offsets_seconds'];assert offsets[0]==r['ttft_seconds']
        assert all(a<=b for a,b in zip(offsets,offsets[1:]))
        assert r['max_text_event_gap_seconds']==max((b-a for a,b in zip(offsets,offsets[1:])),default=0.)
        assert abs(r['mean_decode_token_interval_seconds']-(r['wall_seconds']-r['ttft_seconds'])/1023)<1e-12
    assert row['total_tokens_per_second']==64*5120/row['duration_seconds']
    assert row['mean_ttft_seconds']==statistics.mean(r['ttft_seconds'] for r in row['responses'])
    config=read(root/'config_preflight_not_runtime_owner.json');assert config['use_v2_model_runner'] is True
    reason=read(root/'stop_reason.json');assert reason['original_state_not_rewritten'] and reason['complete'] is False
    assert not (root/'comparison.json').exists()
    report=dict(complete=True,interrupted_not_successful_experiment=True,candidate_not_run=True,
                all69_frozen_snapshots_and_owned_sources_verified=True,all64_control_responses_recomputed=True,
                recorded_guard_no_hits=True,fresh_configuration_preflight_not_runtime_owner=True,
                control_round1_tok_s=row['total_tokens_per_second'],adopted=False)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
