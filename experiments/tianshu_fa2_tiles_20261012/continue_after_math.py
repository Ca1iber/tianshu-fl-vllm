# SPDX-License-Identifier: Apache-2.0
"""Audit the ongoing, independent tile Math run, then run official 4K once.

This controller never modifies model/evaluation sources or replaces services.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def main():
    base = Path('/workspace/benchmark-results/20261012_fa2_tiles_tianshu')
    math = base / 'math_tile128'
    official = Path('/workspace/benchmark-results/20261012_fa2_tile128_official4k_tianshu')
    script = Path(__file__).resolve().parent
    lease = Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    stop = datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
    state_file = base / 'continuation.json'
    if state_file.exists():
        raise RuntimeError('Continuation already exists; inspect it rather than duplicate work')
    state = {'phase': 'waiting_for_math', 'started_utc': datetime.now(timezone.utc).isoformat()}

    def save():
        state_file.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)

    save()
    while not (math / 'result.json').exists():
        if datetime.now(timezone.utc) >= stop:
            state['phase'] = 'deadline_reached'
            save()
            return 1
        time.sleep(3)
    result = json.loads((math / 'result.json').read_text())
    if result['phase'] != 'evaluation_finished_pending_report_audit':
        state.update(phase='math_failed', math_phase=result['phase'])
        save()
        return 1
    subprocess.run([sys.executable, str(script.parent / 'tianshu_adaptive_decode_20261012/audit_math_eval.py'),
                    '--output', str(math)], check=True)
    audit = json.loads((math / 'accuracy_audit.json').read_text())
    archive = Path('/tmp/tianshu_fa2_tile_math_complete.tgz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(math, arcname=math.name)
    data = archive.read_bytes()
    parts = []
    for index, offset in enumerate(range(0, len(data), 262144)):
        part = archive.with_name(archive.name + f'.part{index:03d}')
        chunk = data[offset:offset + 262144]
        part.write_bytes(chunk)
        parts.append({'name': part.name, 'bytes': len(chunk), 'sha256': hashlib.sha256(chunk).hexdigest()})
    manifest = {'archive': str(archive), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'parts': parts}
    Path('/tmp/tianshu_fa2_tile_math_complete_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    state.update(phase='math_audited', math_correct=audit['correct'], math_total=audit['total'],
                 requirement_passed=audit['official_requirement_passed'], backup=manifest)
    save()
    if not audit['official_requirement_passed']:
        state['phase'] = 'quality_gate_failed'
        save()
        return 1
    if official.exists() and (official / 'state.json').exists():
        raise RuntimeError('Official trial already exists; refusing duplication')
    command = [sys.executable, '-u', str(script / 'run_official_4k.py'),
               '--output', str(official), '--control-trial', str(base / 'refill4k1k_128/tile_0'),
               '--candidate-trial', str(base / 'refill4k1k_128/tile_1'),
               '--math-results', str(math), '--max-batched-tokens', '2048']
    state.update(phase='official_running', official_command=command)
    save()
    code = subprocess.call(command)
    state.update(phase='official_finished' if code == 0 else 'official_failed', official_exit_code=code,
                 finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
