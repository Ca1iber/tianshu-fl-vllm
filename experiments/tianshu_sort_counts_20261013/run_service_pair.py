# SPDX-License-Identifier: Apache-2.0
"""Serialized, lease-bounded service counter comparison after exact checks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--micro-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    micro = json.loads((args.micro_root / 'result.json').read_text())
    assert micro['complete'] and len(micro['histogram_cases']) == 16 and len(micro['sort_cases']) == 120
    assert all(row['exact_cpu_histogram_passed'] for case in micro['histogram_cases'] for row in case['variants'])
    assert all(row.get('actual_histogram_routed') and row.get('values_exact')
               and row.get('stable_indices_exact') and row.get('graph_values_exact')
               and row.get('graph_stable_indices_exact') and row.get('indices_dtype') == 'torch.int64'
               and 'error' not in row for row in micro['sort_cases'])
    assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
    stop = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    remaining = int((stop - datetime.now(timezone.utc)).total_seconds())
    assert remaining >= 2400
    signal.alarm(remaining)
    state = dict(phase='running', runner_pid=os.getpid(), official_benchmark=False,
                 adopted=False, requests_per_round=64, rounds=5, excluded_rounds=[1, 2],
                 input_tokens=4096, output_tokens=1024, concurrency=64)
    child = None

    class StopRequested(BaseException):
        pass

    def stop_handler(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise StopRequested(f'Signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, stop_handler)

    def save():
        temporary = root / 'state.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(root / 'state.json')

    save()
    try:
        trial = Path(__file__).resolve().parents[1] / 'tianshu_native_prefill_20261012/run_service_trial.py'
        (root / 'run_service_trial.py').write_bytes(trial.read_bytes())
        (root / 'run_service_pair.py').write_bytes(Path(__file__).read_bytes())
        for name, bits in [('control', 'off'), ('candidate', '32')]:
            state['current_job'] = name
            command = ['python3', '-u', str(trial), '--output', str(root / name),
                       '--buckets', '1', '--gather', '1', '--bucket-trace', '2',
                       '--hybrid', '1', '--adaptive', '1', '--tile', '1', '--native-prefill', '1',
                       '--decode-launch', '1', '--max-batched-tokens', '2048',
                       '--hist-count-bits', bits, '--requests', '64', '--rounds', '5']
            with (root / f'{name}_runner.log').open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state['child_pid'] = child.pid
                save()
                assert child.wait() == 0
            assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
        control = json.loads((root / 'control/result.json').read_text())
        candidate = json.loads((root / 'candidate/result.json').read_text())
        assert control['phase'] == candidate['phase'] == 'complete'
        for key in ('workload', 'installed_sources', 'decode_hook', 'tile_hook', 'bucket_trace_hook'):
            assert control[key] == candidate[key], key
        assert control['histogram_count_bits'] == 'off' and candidate['histogram_count_bits'] == '32'
        assert control['watermark'] == candidate['watermark'] == 0
        assert control['decode_graph_max'] == candidate['decode_graph_max'] == 0
        for left, right in zip(control['short_4k_rounds'], candidate['short_4k_rounds']):
            assert left['successful_requests'] == right['successful_requests'] == 64
            assert [r['seed'] for r in left['responses']] == [r['seed'] for r in right['responses']]
            assert all(r['usage']['prompt_tokens'] == 4096 and r['usage']['completion_tokens'] == 1024
                       for group in (left, right) for r in group['responses'])
        assert all(left['text'] == right['text'] for left, right in zip(control['rare_cases'], candidate['rare_cases']))
        scores = [statistics.mean(r['total_tokens_per_second'] for r in d['short_4k_rounds'][2:])
                  for d in (control, candidate)]
        summary = dict(complete=True, official_benchmark=False, adopted=False, excluded_rounds=[1, 2],
                       control_total_tok_s=scores[0], candidate_total_tok_s=scores[1],
                       gain_percent=100 * (scores[1] / scores[0] - 1),
                       control_rounds=[r['total_tokens_per_second'] for r in control['short_4k_rounds']],
                       candidate_rounds=[r['total_tokens_per_second'] for r in candidate['short_4k_rounds']],
                       actual_count_routes=candidate['sort_count_routing_records'],
                       independent_math_not_run=True, new_original_official_not_run=True)
        (root / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
        state.update(phase='complete', comparison=summary)
    except StopRequested as error:
        state.update(phase='stopped', error=str(error))
    except Exception as error:
        state.update(phase='failed', error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        save()
        (root / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
    if state['phase'] == 'complete':
        subprocess.run(['python3', str(Path(__file__).resolve().parents[1] / 'archive_completed_pair.py'),
                        '--single-probe', '--source', str(root), '--label', 'sort_count32_service_20261013'], check=True)
    return 0 if state['phase'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
