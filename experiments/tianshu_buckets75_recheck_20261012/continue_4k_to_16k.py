# SPDX-License-Identifier: Apache-2.0
"""Explicitly requested original16K after quality/original4K and GPU cleanup."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    previous = args.previous_root.resolve()
    base = Path(__file__).resolve().parents[1]
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    state = dict(phase='waiting_for_complete4k', runner_pid=os.getpid(), jobs=[],
                 workload=[16384, 1024, 64, 128], official_started=False)

    def save():
        p = root / 'state.tmp'
        p.write_text(json.dumps(state, indent=2) + '\n')
        p.replace(root / 'state.json')

    def run(name, command):
        assert datetime.now(timezone.utc) < deadline
        state.update(phase='running', current_job=name)
        save()
        with (root / (name + '_runner.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    cwd='/workspace/vllm-plugin-FL')
        state['jobs'].append(dict(name=name, exit_code=result.returncode))
        save()
        assert result.returncode == 0, name + ' failed'

    save()
    try:
        while not (previous / 'result.json').exists():
            assert datetime.now(timezone.utc) < deadline
            time.sleep(5)
        assert json.loads((previous / 'result.json').read_text())['phase'] == 'complete'
        assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
        assert (deadline - datetime.now(timezone.utc)).total_seconds() > 14400
        quality = previous / 'math'
        audit = json.loads((quality / 'accuracy_audit.json').read_text())
        assert audit['accuracy'] > .95 and audit['buckets'] == audit['gather'] == '1'
        state['official_started'] = True
        save()
        run('official16k', ['python3', str(base / 'tianshu_native_prefill_20261012/run_official_16k.py'),
                            '--output', str(root / 'official'), '--math-results', str(quality),
                            '--control-trial', '/workspace/benchmark-results/20261012_decode_stage1_buckets75_trace_tianshu/candidate',
                            '--candidate-trial', '/workspace/benchmark-results/20261012_decode_stage1_buckets75_gather_tianshu/candidate',
                            '--decode-launch', '1', '--max-batched-tokens', '2048',
                            '--buckets', '1', '--gather', '1', '--bucket-trace', '2', '--requested-benchmark'])
        run('audit16k', ['python3', str(base / 'tianshu_native_prefill_20261012/audit_official_16k.py'),
                         '--output', str(root / 'official'), '--math-results', str(quality),
                         '--previous-best-tok-s', '915.15', '--previous-best-ttft-ms', '599262.03'])
        run('archive16k', ['python3', str(base / 'archive_completed_pair.py'), '--source', str(root / 'official'),
                            '--single-probe', '--label', 'buckets75_gather_official16k_20261013'])
        state['phase'] = 'complete'
    except Exception as error:
        state.update(phase='failed', error=str(error), error_type=type(error).__name__)
    save()
    (root / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
    return 0 if state['phase'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
