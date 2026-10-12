# SPDX-License-Identifier: Apache-2.0
"""Run the native GQA prefill tile sweep before the M8 followups."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('/workspace/benchmark-results/20261012_native_prefill_tianshu')
    root.mkdir(exist_ok=True)
    state_file = root / 'controller_state.json'
    assert not state_file.exists()
    state = {'phase': 'waiting_for_prefill_gemm_pair', 'started_utc': datetime.now(timezone.utc).isoformat()}

    def save():
        state_file.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)

    save()
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    dependency = Path('/workspace/benchmark-results/20261012_prefill_gemm_tianshu/refill128/comparison.json')
    while not dependency.exists():
        if (deadline - datetime.now(timezone.utc)).total_seconds() < 2700:
            state['phase'] = 'failed'
            state['error'] = 'Insufficient remaining lease'
            save()
            return 1
        time.sleep(5)
    assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
    directory = Path(__file__).resolve().parent
    out = root / 'numerical_probe'
    command = [sys.executable, '-u', str(directory / 'probe_prefill.py'), '--output', str(out)]
    state.update(phase='running', command=command)
    save()
    with (root / 'probe.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    state.update(phase='complete' if code == 0 else 'failed', exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    if code == 0:
        subprocess.run([sys.executable, str(directory.parent / 'archive_completed_pair.py'),
                        '--source', str(out), '--label', 'native_prefill_numerical_20261012', '--single-probe'], check=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
