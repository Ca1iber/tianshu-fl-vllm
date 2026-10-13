# SPDX-License-Identifier: Apache-2.0
"""Fresh normal16K/1K N64 service comparison for Native stages1/2."""
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
    parser.add_argument('--archive-label',default='prefill_stage2_normal16k_service_20261013')
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    micro = json.loads((args.micro_root / 'result.json').read_text())
    assert micro['complete'] and micro['numerical_checks_passed'] and micro['all_cases_bitwise_equal'] and len(micro['cases'])==8
    assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
    stop = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    remaining = int((stop - datetime.now(timezone.utc)).total_seconds())
    assert remaining >= 3600
    signal.alarm(remaining)
    state = dict(phase='running', runner_pid=os.getpid(), official_benchmark=False,
                 adopted=False, requests_per_round=64, rounds=3, excluded_rounds=[1],
                 input_tokens=16384, output_tokens=1024, concurrency=64)
    state.update(control_reused=False,control_root=str(root/'control'))
    child = None

    class StopRequested(BaseException):
        pass

    def stop_handler(signum, frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        signal.signal(signal.SIGINT,signal.SIG_IGN)
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
        (root / 'run_stage2_normal16k_pair.py').write_bytes(Path(__file__).read_bytes())
        jobs=[('control','1'),('candidate','2')]
        for name, enabled in jobs:
            state['current_job'] = name
            command = ['python3', '-u', str(trial), '--output', str(root / name),
                       '--buckets', '1', '--gather', '1', '--bucket-trace', '2',
                       '--hybrid', '1', '--adaptive', '1', '--tile', '1', '--native-prefill', '1',
                       '--decode-launch', '1', '--max-batched-tokens', '2048',
                       '--hist-count-bits', 'off', '--address32', '1', '--prefill-stages', enabled, '--input-tokens', '16384', '--output-tokens', '1024', '--concurrency', '64', '--request-timeout', '3600', '--requests', '64', '--rounds', '3']
            with (root / f'{name}_runner.log').open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state['child_pid'] = child.pid
                save()
                assert child.wait() == 0
            assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
        control = json.loads((root/'control/result.json').read_text())
        candidate = json.loads((root / 'candidate/result.json').read_text())
        assert control['phase'] == candidate['phase'] == 'complete'
        for key in ('workload', 'installed_sources', 'decode_hook', 'tile_hook', 'bucket_trace_hook'):
            assert control[key] == candidate[key], key
        assert control['histogram_count_bits'] == candidate['histogram_count_bits'] == 'off'
        assert control['address32']==candidate['address32']=='1'
        assert control['prefill_stages']==1 and candidate['prefill_stages']==2
        assert control['address32_sources']==candidate['address32_sources']
        assert control['watermark'] == candidate['watermark'] == 0
        assert control['decode_graph_max'] == candidate['decode_graph_max'] == 0
        for left, right in zip(control['short_4k_rounds'], candidate['short_4k_rounds']):
            assert left['successful_requests'] == right['successful_requests'] == 64
            assert [r['seed'] for r in left['responses']] == [r['seed'] for r in right['responses']]
            assert all(r['usage']['prompt_tokens'] == 16384 and r['usage']['completion_tokens'] == 1024
                       for group in (left, right) for r in group['responses'])
        assert all(left['text'] == right['text'] for left, right in zip(control['rare_cases'], candidate['rare_cases']))
        scores = [statistics.mean(r['total_tokens_per_second'] for r in d['short_4k_rounds'][1:])
                  for d in (control, candidate)]
        summary = dict(complete=True, official_benchmark=False, adopted=False, excluded_rounds=[1],
                       control_reused=False,control_root=str(root/'control'),
                       workload=dict(input_tokens=16384,output_tokens=1024,requests_per_round=64,concurrency=64),
                       control_total_tok_s=scores[0], candidate_total_tok_s=scores[1],
                       gain_percent=100 * (scores[1] / scores[0] - 1),
                       control_rounds=[r['total_tokens_per_second'] for r in control['short_4k_rounds']],
                       candidate_rounds=[r['total_tokens_per_second'] for r in candidate['short_4k_rounds']],
                       actual_prefill_stage2_records=candidate['prefill_stage2_configuration_records'],
                       actual_prefill_routes=candidate['native_prefill_routing_records'],
                       actual_address_routes=candidate['address32_routing_records'],
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
                        '--single-probe', '--source', str(root), '--label', args.archive_label], check=True)
    return 0 if state['phase'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
