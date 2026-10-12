# SPDX-License-Identifier: Apache-2.0
"""Matched C64 refill test after the owned small-GEMM pair exits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for-small-pair', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'controller_state.json'
    if state_path.exists():
        raise RuntimeError('Refusing duplicate sampling experiment')
    state = {'phase': 'preconditions', 'runner_pid': __import__('os').getpid(),
             'started_utc': datetime.now(timezone.utc).isoformat()}

    def save():
        state_path.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)

    save()
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    if args.wait_for_small_pair:
        state['phase'] = 'waiting_for_small_gemm_cleanup'
        save()
        pair = Path('/workspace/benchmark-results/20261012_small_gemm_tianshu/service_c8')
        while not (pair / 'comparison.json').exists():
            if datetime.now(timezone.utc) >= deadline:
                state['phase'] = 'deadline_reached'
                save()
                return 1
            time.sleep(5)
    numerical = Path('/workspace/benchmark-results/20261012_next_diagnostics_tianshu/sampler_v2_cpu/result.json')
    data = json.loads(numerical.read_text())
    assert data['complete'] and data['cases_passed'] == data['cases_total'] == 105
    kernel = Path(__file__).with_name('experimental_sampler.py')
    assert hashlib.sha256(kernel.read_bytes()).hexdigest() == data['source_sha256']
    assert (deadline - datetime.now(timezone.utc)).total_seconds() > 1500
    results = []
    for flag in ('0', '1'):
        assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
        out = root / f'sampler_{flag}'
        out.mkdir(exist_ok=True)
        assert not (out / 'state.json').exists()
        command = [sys.executable, '-u', str(Path(__file__).with_name('run_service_trial.py')),
                   '--output', str(out), '--sampler-fusion', flag,
                   '--buckets', '0', '--gather', '0', '--tile', '1', '--hybrid', '1',
                   '--adaptive', '1', '--max-batched-tokens', '2048',
                   '--requests', '192', '--concurrency', '64', '--output-tokens', '128', '--rounds', '3']
        state.update(phase='running', current_flag=flag, command=command)
        save()
        with (out / 'runner.log').open('w') as log:
            r = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads((out / 'result.json').read_text())
        if r.returncode or result['phase'] != 'complete':
            state.update(phase='failed', error=result.get('error'), exit_code=r.returncode)
            save()
            return 1
        results.append(result)
    assert results[0]['workload'] == results[1]['workload']
    assert results[0]['installed_sources'] == results[1]['installed_sources']
    means = [statistics.mean(x['total_tokens_per_second'] for x in r['short_4k_rounds'][1:]) for r in results]
    ttfts = [statistics.mean(x['mean_ttft_seconds'] * 1000 for x in r['short_4k_rounds'][1:]) for r in results]
    comparison = {
        'official_benchmark': False, 'independent_quality_pending': True,
        'baseline_tok_s': means[0], 'candidate_tok_s': means[1],
        'throughput_gain_percent': 100 * (means[1] / means[0] - 1),
        'baseline_ttft_ms': ttfts[0], 'candidate_ttft_ms': ttfts[1],
        'ttft_change_percent': 100 * (ttfts[1] / ttfts[0] - 1),
        'workload': results[0]['workload'],
        'greedy_probes_matched': all(a['text'] == b['text'] for a, b in zip(results[0]['rare_cases'], results[1]['rare_cases'])),
        'greedy_probe_count': len(results[0]['rare_cases']),
        'greedy_temperature_zero_probes_exercise_sampling_fusion': False,
        'requests_successful': sum(len(r['rare_cases']) + sum(x['successful_requests'] for x in r['short_4k_rounds']) for r in results),
        'actual_sampler_records': results[1]['sampler_routing_records']}
    (root / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    state.update(phase='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
