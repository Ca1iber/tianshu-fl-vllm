# SPDX-License-Identifier: Apache-2.0
"""Resume owned experiments serially after native-prefill formal verification."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


BASE = Path('/workspace/benchmark-results')
SCRIPTS = Path(__file__).resolve().parent
LEASE = BASE / '20261011_121020_baseline_tianshu/reservation_deadline.json'
NATIVE = BASE / '20261012_native_prefill_tianshu'


def remaining():
    # Re-read the lease: the user can extend it while this controller waits.
    deadline = datetime.fromisoformat(json.loads(LEASE.read_text())['stop_at_utc'])
    return (deadline - datetime.now(timezone.utc)).total_seconds()


def idle():
    return 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)


def owned(pid, script):
    p = Path(f'/proc/{pid}/cmdline')
    return p.exists() and str(script).encode() in p.read_bytes().split(b'\0')


def main():
    root = BASE / '20261012_extended_serial_tianshu'
    root.mkdir(exist_ok=True)
    state_file = root / 'state.json'
    assert not state_file.exists(), 'Refusing to overwrite an existing queue'
    state = dict(phase='waiting_for_native_formal_cleanup', jobs=[],
                 started_utc=datetime.now(timezone.utc).isoformat())

    def save():
        state['remaining_seconds'] = int(remaining())
        tmp = root / 'state.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(state_file)
        print(json.dumps(state), flush=True)

    save()
    terminal = {'quality_requirement_failed', 'math_failed', 'official_failed',
                'official_complete_pending_audit', 'official_not_started_insufficient_lease'}
    while True:
        f = NATIVE / 'continuation_state.json'
        native_state = json.loads(f.read_text()) if f.exists() else {}
        if native_state.get('phase') in terminal and idle():
            # A completed formal trial must have its independent audit before
            # this queue can alter the GPU workload.
            if native_state['phase'] == 'official_complete_pending_audit':
                audit = NATIVE / 'official4k/comparison.json'
                if not audit.exists() or not json.loads(audit.read_text()).get('complete'):
                    time.sleep(5)
                    continue
            elif native_state['phase'] == 'official_not_started_insufficient_lease':
                state['phase'] = 'manual_official_start_required'
                save()
                return 1
            break
        if remaining() < 1800:
            state['phase'] = 'insufficient_remaining_lease'
            save()
            return 1
        time.sleep(5)

    # These controllers have cached the old noon deadline. Replace only the
    # verified own waiting processes; their raw logs/state remain intact.
    for pid, rel in [(93491, 'tianshu_small_gemm_20261012/run_followups.py'),
                     (90223, 'tianshu_fa2_tiles_20261012/run_extended_after_followups.py'),
                     (92415, 'tianshu_decode_mapping_20261012/run_after_tiles.py')]:
        script = SCRIPTS / rel
        if owned(pid, script):
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
            state.setdefault('replaced_waiters', []).append({'pid': pid, 'script': str(script)})
    save()

    held = BASE / '20261012_small_gemm_tianshu/service_c64_4k1k'
    script = SCRIPTS / 'tianshu_small_gemm_20261012/run_pair.py'
    assert owned(94130, script), 'The held pair requires manual recovery'
    assert idle()
    status = Path('/proc/94130/status').read_text()
    assert '\nState:\tT' in status, 'Expected the explicitly held parent'
    state.update(phase='running', current_job='held_m8_c64_candidate')
    save()
    os.kill(94130, signal.SIGCONT)
    while owned(94130, script):
        if remaining() <= 0:
            state['phase'] = 'deadline_reached'
            save()
            return 1
        time.sleep(5)
    assert idle()
    comparison = held / 'comparison.json'
    if not comparison.exists():
        state['phase'] = 'held_pair_failed'
        save()
        return 1
    state['jobs'].append(dict(name='held_m8_c64_candidate', exit_code=0, output=str(held)))
    subprocess.run([sys.executable, str(SCRIPTS/'archive_completed_pair.py'),
                    '--source', str(held), '--label', 'small_gemm_service_c64_4k1k_20261012'], check=True)

    jobs = [
        ('prefill_gemm_standard_confirmation', 2700,
         BASE/'20261012_prefill_gemm_tianshu/service_c64_4k1k',
         [str(SCRIPTS/'tianshu_prefill_gemm_20261012/confirm_with_shared_control.py'),
          '--control', str(held/'small_0')], None),
        ('m8_c8_warmed_confirmation', 2700,
         BASE/'20261012_small_gemm_tianshu/service_c8_warmed',
         [str(SCRIPTS/'tianshu_small_gemm_20261012/run_pair.py'),
          '--concurrency', '8', '--requests', '16', '--warmup-rounds', '3'],
         'small_gemm_service_c8_warmed_20261012'),
        ('direct_decode_mapping_numerical', 1800,
         BASE/'20261012_decode_mapping_tianshu/numerical_probe',
         [str(SCRIPTS/'tianshu_decode_mapping_20261012/probe_mapping.py')],
         'decode_mapping_numerical_20261012'),
    ]
    for name, reserve, output, command, label in jobs:
        if remaining() < reserve:
            state.update(phase='insufficient_remaining_lease', pending_job=name)
            save()
            return 1
        assert idle() and not output.exists(), (name, output)
        state.update(phase='running', current_job=name)
        save()
        with (root/f'{name}.log').open('w') as log:
            result = subprocess.run([sys.executable, '-u', *command, '--output', str(output)],
                                    stdout=log, stderr=subprocess.STDOUT)
        assert idle(), f'GPU cleanup failed after {name}'
        state['jobs'].append(dict(name=name, exit_code=result.returncode, output=str(output)))
        save()
        if result.returncode:
            state.update(phase='failed', failed_job=name)
            save()
            return 1
        if label:
            archive = [sys.executable, str(SCRIPTS/'archive_completed_pair.py'),
                       '--source', str(output), '--label', label]
            if name.endswith('numerical'):
                archive.append('--single-probe')
            subprocess.run(archive, check=True)
    state.update(phase='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
