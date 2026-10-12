# SPDX-License-Identifier: Apache-2.0
"""Lease-bounded, serial transport experiment after official and fusion cleanup."""
import argparse
from datetime import datetime, timezone
import hashlib
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
    experiments = Path(__file__).resolve().parents[1]
    results = Path('/workspace/benchmark-results')
    previous = results / '20261012_decode_launch_tianshu/followup_tile16'
    fusion = results / '20261012_gateup_silu_tianshu/affine_numerical/result.json'
    deadline = datetime.fromisoformat(json.loads((results /
        '20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    state = dict(phase='waiting', runner_pid=os.getpid(), official_benchmark=False,
                 accuracy_test=False, adoption_allowed=False, jobs=[])

    def remaining():
        return (deadline - datetime.now(timezone.utc)).total_seconds()

    def save():
        p = root / 'state.tmp'
        p.write_text(json.dumps(state, indent=2) + '\n')
        p.replace(root / 'state.json')

    def idle():
        return 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)

    save()
    try:
        while True:
            assert remaining() > 0, 'Lease ended while waiting'
            if (previous / 'result.json').exists() and fusion.exists():
                parent = json.loads((previous / 'result.json').read_text())
                numerical = json.loads(fusion.read_text())
                if parent.get('phase') == 'complete' and numerical.get('complete') and idle():
                    break
                if parent.get('phase') == 'failed':
                    raise RuntimeError('Previous official chain failed; root must diagnose')
            time.sleep(10)
        if remaining() < 2400:
            state.update(phase='skipped', reason='Less than40min for serial paired service trial')
        else:
            comparison = json.loads((previous / 'official/comparison.json').read_text())
            use_stage1 = bool(comparison['adopt_as_best'])
            state.update(decode_launch='1' if use_stage1 else '0',
                         prerequisite_official=comparison, phase='running')
            trial = experiments / 'tianshu_native_prefill_20261012/run_service_trial.py'
            (root / 'run_service_trial.py').write_bytes(trial.read_bytes())
            state['trial_sha256'] = hashlib.sha256(trial.read_bytes()).hexdigest()
            save()
            for name, interval in [('control', 1), ('candidate', 4)]:
                assert idle() and remaining() > 900
                command = ['python3', str(trial), '--output', str(root / name),
                           '--buckets', '0', '--gather', '0', '--hybrid', '1',
                           '--adaptive', '1', '--tile', '1', '--native-prefill', '1',
                           '--decode-launch', state['decode_launch'],
                           '--max-batched-tokens', '2048', '--output-tokens', '1024',
                           '--requests', '64', '--concurrency', '64', '--rounds', '3',
                           '--stream-interval', str(interval)]
                state['current_job'] = name
                save()
                with (root / (name + '_runner.log')).open('w') as log:
                    run = subprocess.run(command, cwd='/workspace/vllm-plugin-FL',
                                         stdout=log, stderr=subprocess.STDOUT)
                state['jobs'].append(dict(name=name, interval=interval, exit_code=run.returncode))
                save()
                assert run.returncode == 0, name + ' failed'
            pair = [json.loads((root / name / 'result.json').read_text())
                    for name in ('control', 'candidate')]
            for experiment in pair:
                assert experiment['phase'] == 'complete'
                assert len(experiment['short_4k_rounds']) == 3
                for run in experiment['short_4k_rounds']:
                    assert run['successful_requests'] == 64
                    for response in run['responses']:
                        assert response['usage']['prompt_tokens'] == 4096
                        assert response['usage']['completion_tokens'] == 1024
            # Both groups use the existing identical prompt seed recipe.
            for a, b in zip(pair[0]['short_4k_rounds'], pair[1]['short_4k_rounds']):
                assert [x['seed'] for x in a['responses']] == [x['seed'] for x in b['responses']]
            values = [[x['total_tokens_per_second'] for x in s['short_4k_rounds'][1:]] for s in pair]
            means = list(map(statistics.mean, values))
            ttft = [1000 * statistics.mean(x['mean_ttft_seconds'] for x in s['short_4k_rounds'][1:])
                    for s in pair]
            tuning = [[len(x['tuning_records']) for x in s['short_4k_rounds']] for s in pair]
            gain = 100 * (means[1] / means[0] - 1)
            clean = all(not any(x[1:]) for x in tuning)
            summary = dict(complete=True, official_benchmark=False, adoption_allowed=False,
                           workload=dict(input=4096, output=1024, concurrency=64,
                                         requests=64, rounds=3, excluded_rounds=[1]),
                           control_stream_interval=1, candidate_stream_interval=4,
                           control_total_tok_s=means[0], candidate_total_tok_s=means[1],
                           gain_percent=gain, control_ttft_ms=ttft[0], candidate_ttft_ms=ttft[1],
                           tuning=tuning, measured_tuning_clean=clean,
                           requests_successful=384,
                           promising=clean and gain > 5 and min(values[1]) > max(values[0]),
                           independent_accuracy_not_run=True, original_official_not_run=True)
            (root / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
            state.update(phase='complete', comparison=summary)
    except Exception as error:
        state.update(phase='failed', error=str(error), error_type=type(error).__name__)
    save()
    (root / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
    return 0 if state['phase'] in ('complete', 'skipped') else 1


if __name__ == '__main__':
    raise SystemExit(main())
