# SPDX-License-Identifier: Apache-2.0
"""Wait for the owned budget pair cleanup, then run one metadata probe."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    pair=Path('/workspace/benchmark-results/20261012_fa2_budget4096_tianshu/prefill_short')
    root=Path('/workspace/benchmark-results/20261012_metadata_fusion_tianshu')
    root.mkdir(parents=True,exist_ok=True)
    state_file=root/'controller_state.json'
    assert not state_file.exists(), 'Refusing duplicate metadata probe'
    state={'phase':'waiting_for_budget_pair','started_utc':datetime.now(timezone.utc).isoformat()}
    def save():
        state_file.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    save()
    stop=datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    while not (pair/'comparison.json').exists():
        if datetime.now(timezone.utc)>=stop:
            state['phase']='deadline_reached';save();return 1
        time.sleep(3)
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    command=[sys.executable,'-u',str(Path(__file__).with_name('probe_metadata.py')),
             '--output',str(root/'integer_probe')]
    with (root/'probe.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='running',pid=child.pid,command=command);save();code=child.wait()
    state.update(phase='complete' if code==0 else 'failed',exit_code=code,
                 finished_utc=datetime.now(timezone.utc).isoformat());save()
    return code


if __name__=='__main__':raise SystemExit(main())
