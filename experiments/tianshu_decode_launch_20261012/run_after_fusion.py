# SPDX-License-Identifier: Apache-2.0
"""Run the numerical fusion probe only after prior owned GPU work is idle."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    base=Path('/workspace/benchmark-results')
    root=base/'20261012_decode_launch_tianshu'
    root.mkdir(exist_ok=True)
    state_file=root/'controller_state.json'
    assert not state_file.exists()
    lease=base/'20261011_121020_baseline_tianshu/reservation_deadline.json'
    state=dict(phase='waiting_for_gateup_fusion_cleanup',started_utc=datetime.now(timezone.utc).isoformat())

    def remaining():
        deadline=datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
        return (deadline-datetime.now(timezone.utc)).total_seconds()

    def save():
        state['remaining_seconds']=int(remaining())
        temporary=root/'state.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(state_file)
        print(json.dumps(state),flush=True)

    save()
    dependency=base/'20261012_gateup_silu_tianshu/controller_state.json'
    while True:
        d=json.loads(dependency.read_text()) if dependency.exists() else {}
        if d.get('phase') in {'complete','failed'}:
            if 'No running processes found' in subprocess.check_output(['ixsmi'],text=True):
                break
        if remaining()<1800:
            state['phase']='insufficient_remaining_lease';save();return 1
        time.sleep(5)
    directory=Path(__file__).resolve().parent
    output=root/'numerical_probe'
    state.update(phase='running',output=str(output));save()
    with (root/'probe.log').open('w') as log:
        result=subprocess.run([sys.executable,'-u',str(directory/'probe_launch.py'),'--output',str(output)],
                              stdout=log,stderr=subprocess.STDOUT)
    state.update(phase='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,
                 finished_utc=datetime.now(timezone.utc).isoformat());save()
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    if result.returncode==0:
        subprocess.run([sys.executable,str(directory.parent/'archive_completed_pair.py'),
                        '--source',str(output),'--label','decode_launch_numerical_20261012','--single-probe'],check=True)
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
