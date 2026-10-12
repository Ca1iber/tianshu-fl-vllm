# SPDX-License-Identifier: Apache-2.0
"""Profile a formally improved native-prefill profile, then isolate budget4096."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    base = Path('/workspace/benchmark-results')
    root = base/'20261012_native_prefill_followups_tianshu'
    root.mkdir(exist_ok=True)
    state_file = root/'state.json'
    assert not state_file.exists()
    lease = base/'20261011_121020_baseline_tianshu/reservation_deadline.json'
    state = dict(phase='waiting_for_extended_serial_queue', jobs=[],
                 started_utc=datetime.now(timezone.utc).isoformat())

    def remaining():
        deadline = datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
        return (deadline-datetime.now(timezone.utc)).total_seconds()

    def idle():
        return 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)

    def save():
        state['remaining_seconds'] = int(remaining())
        tmp = root/'state.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(state_file)
        print(json.dumps(state), flush=True)

    save()
    dependency = base/'20261012_extended_serial_tianshu/state.json'
    while True:
        d = json.loads(dependency.read_text()) if dependency.exists() else {}
        if d.get('phase') in {'complete', 'failed', 'held_pair_failed'} and idle():
            break
        if remaining() < 2700:
            state['phase']='insufficient_remaining_lease'
            save()
            return 1
        time.sleep(5)
    native = base/'20261012_native_prefill_tianshu'
    formal = native/'official4k/comparison.json'
    if not formal.exists() or not json.loads(formal.read_text()).get('adopt_as_best'):
        state['phase']='native_profile_not_formally_improved'
        save()
        return 0
    directory = Path(__file__).resolve().parent
    jobs = [
        ('native_c64_warmed_profile', root/'warmed_profile_c64',
         [str(directory.parent/'tianshu_profile_20261012/profile_service.py'),
          '--no-graphs', '--concurrency', '64', '--adaptive', '1', '--tile', '1',
          '--native-prefill', '1', '--max-batched-tokens', '2048', '--warmup-rounds', '2'],
         'native_prefill_warmed_profile_c64_20261012'),
        ('native_budget4096_confirmation', root/'budget4096',
         [str(directory/'run_budget_confirmation.py'), '--budget', '4096',
          '--control', str(native/'service_standard/native_1')], None),
    ]
    for name, output, command, label in jobs:
        assert idle() and not output.exists()
        if remaining() < 2700:
            state.update(phase='insufficient_remaining_lease', pending_job=name)
            save()
            return 1
        state.update(phase='running', current_job=name)
        save()
        with (root/f'{name}.log').open('w') as log:
            process = subprocess.run([sys.executable, '-u', *command, '--output', str(output)],
                                     stdout=log, stderr=subprocess.STDOUT)
        assert idle(), f'GPU cleanup failed after {name}'
        state['jobs'].append(dict(name=name, exit_code=process.returncode, output=str(output)))
        if process.returncode:
            state.update(phase='failed', failed_job=name)
            save()
            return 1
        if label:
            subprocess.run([sys.executable, str(directory.parent/'archive_completed_pair.py'),
                            '--source', str(output), '--label', label, '--single-probe'], check=True)
        save()
    state.update(phase='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
