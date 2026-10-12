# SPDX-License-Identifier: Apache-2.0
"""Lease-bounded fixed-index followup to the measured bucket-only stall."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    (root / 'run_gather_after_bare75.py').write_bytes(Path(__file__).read_bytes())
    base = Path('/workspace/benchmark-results')
    bare = base / '20261012_decode_stage1_buckets75_trace_tianshu'
    deadline = datetime.fromisoformat(json.loads((base /
        '20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    state = dict(phase='waiting', runner_pid=os.getpid(), official_benchmark=False,
                 adoption_allowed=False)

    def remaining():
        return (deadline - datetime.now(timezone.utc)).total_seconds()

    def save():
        p = root / 'state.tmp'
        p.write_text(json.dumps(state, indent=2) + '\n')
        p.replace(root / 'state.json')

    save()
    try:
        while not (bare / 'result.json').exists():
            assert remaining() > 0
            time.sleep(5)
        assert json.loads((bare / 'result.json').read_text())['phase'] == 'complete'
        assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
        if remaining() < 1320:
            state.update(phase='skipped', reason='Less than22min for bounded candidate')
        else:
            n = 128 if remaining() >= 1500 else 64
            control = bare / 'candidate' if n == 128 else base / '20261012_stream_interval_tianshu/control'
            state.update(phase='running', requests=n, control_source=str(control),
                         control_buckets='1' if n == 128 else '0', candidate_buckets='1',
                         control_gather='0', candidate_gather='1')
            save()
            trial = Path('/workspace/vllm-plugin-FL/experiments/tianshu_native_prefill_20261012/run_service_trial.py')
            (root / 'run_service_trial.py').write_bytes(trial.read_bytes())
            command = ['python3', '-u', str(trial), '--output', str(root / 'candidate'),
                       '--buckets', '1', '--gather', '1', '--bucket-trace', '2',
                       '--hybrid', '1', '--adaptive', '1', '--tile', '1', '--native-prefill', '1',
                       '--decode-launch', '1', '--max-batched-tokens', '2048',
                       '--output-tokens', '1024', '--concurrency', '64', '--requests', str(n), '--rounds', '3']
            with (root / 'candidate_runner.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        cwd='/workspace/vllm-plugin-FL')
            assert result.returncode == 0
            before = json.loads((control / 'result.json').read_text())
            after = json.loads((root / 'candidate/result.json').read_text())
            assert before['phase'] == after['phase'] == 'complete'
            assert before['decode_hook'] == after['decode_hook'] and before['tile_hook'] == after['tile_hook']
            a, b = before['short_4k_rounds'][:3], after['short_4k_rounds']
            assert len(a) == len(b) == 3
            for left, right in zip(a, b):
                assert left['successful_requests'] == right['successful_requests'] == n
                assert [x['seed'] for x in left['responses']] == [x['seed'] for x in right['responses']]
                for group in [left, right]:
                    assert all(x['usage']['prompt_tokens'] == 4096 and x['usage']['completion_tokens'] == 1024
                               for x in group['responses'])
            records = root / 'candidate/bucket_trace_records'
            warm = [json.loads(p.read_text()) for p in records.glob('warmup_completed_*.json')]
            assert any(x['complete'] and len(x['calls']) == 75 for x in warm)
            gather = [json.loads(p.read_text()) for p in records.glob('gather_*.json')]
            assert gather and any(x['rows'] == 1 and x['index_dtype'] == 'torch.int32' for x in gather)
            means = [statistics.mean(x['total_tokens_per_second'] for x in rows[1:]) for rows in [a, b]]
            summary = dict(complete=True, official_benchmark=False, adoption_allowed=False,
                           requests_per_round=n, concurrency=64, input_tokens=4096, output_tokens=1024,
                           excluded_rounds=[1], control_reused=True, control_source=str(control),
                           control_buckets=before['buckets'], control_gather=before['gather'],
                           candidate_buckets=after['buckets'], candidate_gather=after['gather'],
                           control_total_tok_s=means[0], candidate_total_tok_s=means[1],
                           gain_percent=100 * (means[1] / means[0] - 1),
                           candidate_tuning_by_round=[len(x['tuning_records']) for x in b],
                           candidate_runtime_tuning=after['runtime_tune_records'],
                           candidate_spread_percent=after['measured_round_stability']['spread_percent'],
                           rare_greedy_matches=[x['text'] == y['text'] for x, y in zip(before['rare_cases'], after['rare_cases'])],
                           actual_gather_records=gather,
                           independent_quality_not_run=True, original_official_not_run=True)
            (root / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
            state.update(phase='complete', comparison=summary)
    except Exception as error:
        state.update(phase='failed', error=str(error), error_type=type(error).__name__)
    save()
    (root / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
    if state['phase'] == 'complete':
        subprocess.run(['python3', '/workspace/vllm-plugin-FL/experiments/archive_completed_pair.py',
                        '--source', str(root), '--single-probe', '--label', 'decode_stage1_buckets75_gather_20261012'], check=True)
    return 0 if state['phase'] in ('complete', 'skipped') else 1


if __name__ == '__main__':
    raise SystemExit(main())
