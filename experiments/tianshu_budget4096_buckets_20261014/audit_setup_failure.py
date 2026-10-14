# SPDX-License-Identifier: Apache-2.0
"""Verify the first budget attempt stopped before candidate server launch."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();pair=root/'budget_service'
    parent=json.loads((root/'result.json').read_text())
    state=json.loads((pair/'result.json').read_text())
    assert parent['phase']==state['phase']=='failed' and state['current_job']=='candidate'
    for directory,item in ((root,parent),(pair,state)):
        for path,digest in item['frozen_sources'].items():
            rel=Path(path).relative_to('/workspace/vllm-plugin-FL/experiments')
            assert hashlib.sha256((directory/'source_snapshot'/rel).read_bytes()).hexdigest()==digest
    control=json.loads((pair/'control/result.json').read_text())
    assert control['phase']=='complete' and control['actual_budget']==2048
    assert control['actual_bucket_points']==75 and control['actual_extended_policy_verified']
    rows=control['short_4k_rounds'];assert len(rows)==3
    for row in rows:
        assert row['successful_requests']==len(row['responses'])==64
        for response in row['responses']:
            assert response['usage']['prompt_tokens']==response['input_tokens']==4096
            assert response['usage']['completion_tokens']==response['output_tokens']==1024
            assert response['usage']['total_tokens']==5120
            assert response['text'] and 0<response['ttft_seconds']<=response['wall_seconds']<=row['duration_seconds']
        assert abs(row['total_tokens_per_second']-64*5120/row['duration_seconds'])<1e-8
        assert abs(row['mean_ttft_seconds']-statistics.mean(v['ttft_seconds'] for v in row['responses']))<1e-9
    log=(pair/'candidate_runner.log').read_text()
    assert 'Bucket confirmation requires the verified native profile at budget2048' in log
    assert 'RuntimeError' in log
    assert not (pair/'candidate/server.log').exists() and not (pair/'candidate/state.json').exists()
    assert not (pair/'comparison.json').exists()
    audit=dict(complete=True,official_benchmark=False,adopted=False,comparison_valid=False,
               scope='Failed attempt: control192 requests valid; candidate blocked by old CLI guard before server',
               original_frozen_sources_snapshots_verified=True,control_successful_requests=192,
               control_rounds=[v['total_tokens_per_second'] for v in rows],candidate_requests=0,
               candidate_server_not_started=True,new_v2_requires_fresh_control=True)
    (root/'local_failure_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
