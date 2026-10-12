# SPDX-License-Identifier: Apache-2.0
"""Wait for the held pair's control cleanup, then run two isolated diagnostics."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('/workspace/benchmark-results/20261012_native_prefill_tianshu')
    state_file = root / 'diagnostic_controller_state.json'
    assert not state_file.exists()
    state = {'phase': 'waiting_for_c64_control_cleanup', 'jobs': [],
             'started_utc': datetime.now(timezone.utc).isoformat()}

    def save():
        state_file.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)

    save()
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    control = Path('/workspace/benchmark-results/20261012_small_gemm_tianshu/service_c64_4k1k/small_0')
    while not (control / 'result.json').exists():
        if (deadline - datetime.now(timezone.utc)).total_seconds() < 900:
            state['phase'] = 'deadline_reached'
            save()
            return 1
        time.sleep(5)
    assert json.loads((control / 'result.json').read_text())['phase'] == 'complete'
    assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
    directory = Path(__file__).resolve().parent
    archive_script = directory.parent / 'archive_completed_pair.py'
    subprocess.run([sys.executable, str(archive_script), '--source', str(control),
                    '--label', 'shared_c64_4k1k_control_20261012', '--single-probe'], check=True)
    for name, script, extra in [('special_math', 'diagnose_math.py', []),
                                ('native_diagnostic', 'probe_prefill.py', ['--diagnostic-only'])]:
        assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
        out = root / name
        command = [sys.executable, '-u', str(directory / script), '--output', str(out)] + extra
        state.update(phase='running', current_job=name, command=command)
        save()
        with (root / f'{name}.log').open('w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        state['jobs'].append({'name': name, 'exit_code': code, 'output': str(out)})
        if code:
            state['phase'] = 'failed'
            save()
            return 1
        subprocess.run([sys.executable, str(archive_script), '--source', str(out),
                        '--label', f'{name}_20261012', '--single-probe'], check=True)
    state.update(phase='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
