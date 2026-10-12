# SPDX-License-Identifier: Apache-2.0
"""Confirm M8 candidate on standard C64 load and a separately warmed C8 pair."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('/workspace/benchmark-results/20261012_small_gemm_tianshu')
    state_path = root / 'followup_state.json'
    assert not state_path.exists()
    state = {'phase': 'waiting_for_native_prefill_probe', 'started_utc': datetime.now(timezone.utc).isoformat(), 'jobs': []}

    def save():
        state_path.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)

    save()
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    dependency = Path('/workspace/benchmark-results/20261012_native_prefill_tianshu/controller_state.json')
    while not (dependency.exists() and json.loads(dependency.read_text()).get('phase') in ('complete', 'failed')):
        if datetime.now(timezone.utc) >= deadline:
            state['phase'] = 'deadline_reached'
            save()
            return 1
        time.sleep(5)
    first = json.loads((root / 'service_c8/comparison.json').read_text())
    assert first['requests_successful'] == 110 and first['greedy_probes_matched']
    assert len(first['actual_small_mm_records']) == 3
    for name, concurrency, requests, warmups in [('service_c64_4k1k', 64, 128, 0), ('service_c8_warmed', 8, 16, 3)]:
        if (deadline - datetime.now(timezone.utc)).total_seconds() < 2700:
            state['phase'] = 'insufficient_remaining_lease'
            save()
            return 1
        assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
        output = root / name
        assert not output.exists()
        command = [sys.executable, '-u', str(Path(__file__).with_name('run_pair.py')),
                   '--output', str(output), '--concurrency', str(concurrency),
                   '--requests', str(requests), '--warmup-rounds', str(warmups)]
        state.update(phase='running', current_job=name)
        save()
        with (root / f'{name}_controller.log').open('w') as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        state['jobs'].append({'name': name, 'exit_code': child.returncode, 'output': str(output)})
        if child.returncode:
            state['phase'] = 'failed'
            save()
            return 1
        archive = [sys.executable, '-u', str(Path(__file__).resolve().parents[1] / 'archive_completed_pair.py'),
                   '--source', str(output), '--label', f'small_gemm_{name}_20261012']
        subprocess.run(archive, check=True)
        if name == 'service_c64_4k1k':
            prefill = json.loads(Path('/workspace/benchmark-results/20261012_prefill_gemm_tianshu/refill128/comparison.json').read_text())
            if prefill['throughput_gain_percent'] > .5 and (deadline - datetime.now(timezone.utc)).total_seconds() > 4500:
                directory = Path(__file__).resolve().parents[1] / 'tianshu_prefill_gemm_20261012'
                prefill_output = Path('/workspace/benchmark-results/20261012_prefill_gemm_tianshu/service_c64_4k1k')
                state.update(phase='running', current_job='prefill_gemm_standard_confirmation')
                save()
                with (root/'prefill_standard_controller.log').open('w') as log:
                    result = subprocess.run([sys.executable,'-u',str(directory/'confirm_with_shared_control.py'),
                        '--output',str(prefill_output),'--control',str(output/'small_0')],stdout=log,stderr=subprocess.STDOUT)
                state['jobs'].append({'name':'prefill_gemm_standard_confirmation','exit_code':result.returncode,'output':str(prefill_output)})
                if result.returncode:
                    state['phase']='failed';save();return 1
    state.update(phase='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
