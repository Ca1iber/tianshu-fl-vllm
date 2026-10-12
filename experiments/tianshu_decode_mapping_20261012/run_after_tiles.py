# SPDX-License-Identifier: Apache-2.0
"""Run a serial native Decode mapping probe after the queued tile sweep."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('/workspace/benchmark-results/20261012_decode_mapping_tianshu')
    root.mkdir(exist_ok=True)
    state_path = root / 'controller_state.json'
    assert not state_path.exists()
    state = {'phase': 'waiting_for_extended_tiles', 'started_utc': datetime.now(timezone.utc).isoformat()}

    def save():
        state_path.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)

    save()
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    dependency = Path('/workspace/benchmark-results/20261012_fa2_extended_tianshu/controller_state.json')
    while True:
        if dependency.exists() and json.loads(dependency.read_text()).get('phase') in ('complete', 'insufficient_remaining_lease'):
            if 'No running processes found' in subprocess.check_output(['ixsmi'], text=True):
                break
        if (deadline - datetime.now(timezone.utc)).total_seconds() < 1800:
            state['phase'] = 'insufficient_remaining_lease'
            save()
            return 1
        time.sleep(5)
    directory = Path(__file__).resolve().parent
    out = root / 'numerical_probe'
    command = [sys.executable, '-u', str(directory / 'probe_mapping.py'), '--output', str(out)]
    state.update(phase='running', command=command)
    save()
    with (root / 'probe.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    state.update(phase='complete' if code == 0 else 'failed', exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    if code == 0:
        subprocess.run([sys.executable, str(directory.parent / 'archive_completed_pair.py'),
                        '--source', str(out), '--label', 'decode_mapping_numerical_20261012', '--single-probe'], check=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
