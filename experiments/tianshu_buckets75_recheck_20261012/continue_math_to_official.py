# SPDX-License-Identifier: Apache-2.0
"""User-requested original4K after matching independent bucket/gather quality."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert root.is_dir() and not (root / 'result.json').exists()
    base = Path(__file__).resolve().parents[1]
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    state = dict(phase='waiting_for_math', runner_pid=os.getpid(), jobs=[], official_started=False)

    def save():
        p = root / 'state.tmp'
        p.write_text(json.dumps(state, indent=2) + '\n')
        p.replace(root / 'state.json')

    def run(name, command):
        assert datetime.now(timezone.utc) < deadline
        state.update(phase='running', current_job=name)
        save()
        with (root / (name + '_runner.log')).open('w') as log:
            result = subprocess.run(command, cwd='/workspace/vllm-plugin-FL',
                                    stdout=log, stderr=subprocess.STDOUT)
        state['jobs'].append(dict(name=name, exit_code=result.returncode))
        save()
        assert result.returncode == 0, name + ' failed'

    save()
    try:
        while not (root / 'math/result.json').exists():
            assert datetime.now(timezone.utc) < deadline
            time.sleep(5)
        quality = root / 'math'
        run('math_audit', ['python3', str(base / 'tianshu_candidate_quality_20261012/audit_math_eval.py'),
                           '--output', str(quality)])
        audit = json.loads((quality / 'accuracy_audit.json').read_text())
        state.update(math_correct=audit['correct'], math_total=105,
                     quality_passed=audit['official_requirement_passed'])
        save()
        run('math_archive', ['python3', str(base / 'archive_completed_pair.py'), '--source', str(quality),
                             '--single-probe', '--completion-file', 'accuracy_audit.json',
                             '--label', 'buckets75_gather_math_20261013'])
        if audit['official_requirement_passed']:
            assert (deadline - datetime.now(timezone.utc)).total_seconds() > 3600
            assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
            official = root / 'official'
            state['official_started'] = True
            save()
            run('official', ['python3', str(base / 'tianshu_native_prefill_20261012/run_official_4k.py'),
                              '--output', str(official), '--math-results', str(quality),
                              '--control-trial', '/workspace/benchmark-results/20261012_decode_stage1_buckets75_trace_tianshu/candidate',
                              '--candidate-trial', '/workspace/benchmark-results/20261012_decode_stage1_buckets75_gather_tianshu/candidate',
                              '--decode-launch', '1', '--max-batched-tokens', '2048',
                              '--buckets', '1', '--gather', '1', '--bucket-trace', '2', '--requested-benchmark'])
            run('official_audit', ['python3', str(base / 'tianshu_native_prefill_20261012/audit_official.py'),
                                    '--output', str(official), '--math-results', str(quality),
                                    '--previous-best-tok-s', '2808.93', '--previous-best-ttft-ms', '4827.51'])
            run('official_archive', ['python3', str(base / 'archive_completed_pair.py'), '--source', str(official),
                                      '--single-probe', '--label', 'buckets75_gather_official_20261013'])
            state['phase'] = 'complete'
        else:
            state.update(phase='quality_failed', reason='Independent accuracy not above95%')
    except Exception as error:
        state.update(phase='failed', error=str(error), error_type=type(error).__name__)
    save()
    (root / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
    return 0 if state['phase'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
