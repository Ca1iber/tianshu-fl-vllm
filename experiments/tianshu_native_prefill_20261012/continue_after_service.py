# SPDX-License-Identifier: Apache-2.0
"""Serial quality and official4K after clear native-prefill service improvement."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('/workspace/benchmark-results/20261012_native_prefill_tianshu')
    state_file = root / 'continuation_state.json'
    assert not state_file.exists()
    state = {'phase':'waiting_for_standard_service','started_utc':datetime.now(timezone.utc).isoformat()}

    def save():
        state_file.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)

    save()
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    pair = root/'service_standard'
    while not (pair/'comparison.json').exists():
        if (deadline-datetime.now(timezone.utc)).total_seconds()<3600:
            state['phase']='insufficient_remaining_lease';save();return 1
        time.sleep(3)
    comparison=json.loads((pair/'comparison.json').read_text())
    if comparison['throughput_gain_percent']<2:
        state.update(phase='no_clear_service_gain',comparison=comparison);save();return 0
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    directory=Path(__file__).resolve().parent
    quality_scripts=directory.parent/'tianshu_candidate_quality_20261012'
    math=root/'math'
    command=[sys.executable,'-u',str(quality_scripts/'run_math_eval.py'),'--output',str(math),'--candidate','native-prefill']
    state.update(phase='math_running',command=command);save()
    with (root/'math_runner.log').open('w') as log:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    if code:
        state.update(phase='math_failed',exit_code=code);save();return 1
    subprocess.run([sys.executable,str(quality_scripts/'audit_math_eval.py'),'--output',str(math)],check=True)
    audit=json.loads((math/'accuracy_audit.json').read_text())
    state.update(phase='math_audited',math_correct=audit['correct'],math_total=105,
                 quality_passed=audit['official_requirement_passed']);save()
    if not audit['official_requirement_passed']:
        state['phase']='quality_requirement_failed';save();return 0
    if (deadline-datetime.now(timezone.utc)).total_seconds()<2100:
        state['phase']='official_not_started_insufficient_lease';save();return 0
    official=root/'official4k'
    command=[sys.executable,'-u',str(directory/'run_official_4k.py'),'--output',str(official),
        '--control-trial',str(pair/'control_0'),'--candidate-trial',str(pair/'native_1'),
        '--max-batched-tokens','2048','--math-results',str(math)]
    state.update(phase='official_running',command=command);save()
    with (root/'official_runner.log').open('w') as log:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    state.update(phase='official_complete_pending_audit' if code==0 else 'official_failed',
                 exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat());save()
    return code


if __name__=='__main__':
    raise SystemExit(main())
