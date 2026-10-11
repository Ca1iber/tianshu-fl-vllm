# SPDX-License-Identifier: Apache-2.0
"""Wait for current formal trial, then serialize profile and independent probes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    source=Path(__file__).resolve().parents[1]
    root=Path('/workspace/benchmark-results/20261012_next_diagnostics_tianshu')
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/'state.json').exists(), 'Refusing duplicate diagnostics'
    continuation=Path('/workspace/benchmark-results/20261012_fa2_tiles_tianshu/continuation.json')
    lease=Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    deadline=datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
    state={'phase':'waiting_for_official','jobs':[],'started_utc':datetime.now(timezone.utc).isoformat()}
    def save():
        (root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    save()
    while True:
        if datetime.now(timezone.utc)>=deadline:
            state['phase']='deadline_reached';save();return 1
        current=json.loads(continuation.read_text())
        if current['phase'] in ('official_finished','official_failed','quality_gate_failed','math_failed'):
            break
        time.sleep(5)
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    quality=bool(current.get('requirement_passed'))
    commands=[
        [str(source/'tianshu_profile_20261012/profile_service.py'),'--output',str(root/'profile_no_graph'),
         '--small-only','--no-graphs','--adaptive','1','--tile','1' if quality else '0','--max-batched-tokens','2048'],
        [str(source/'tianshu_sampler_fusion_20261012/probe_sampler.py'),'--output',str(root/'sampler')],
        [str(source/'tianshu_splitk_20261012/probe_mm.py'),'--output',str(root/'splitk')],
    ]
    for index,args in enumerate(commands):
        if (deadline-datetime.now(timezone.utc)).total_seconds()<180:
            state['phase']='insufficient_time';save();return 1
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
        state['phase']=f'running_{index}';save()
        command=[sys.executable,'-u',*args]
        with (root/f'job_{index}.log').open('w') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['active_pid']=proc.pid;state['active_command']=command;save()
            code=proc.wait()
        state['jobs'].append({'command':command,'script_sha256':hashlib.sha256(Path(args[0]).read_bytes()).hexdigest(),
                              'exit_code':code,'finished_utc':datetime.now(timezone.utc).isoformat()})
        save()
        if 'No running processes found' not in subprocess.check_output(['ixsmi'],text=True):
            raise RuntimeError('GPU cleanup needs inspection before another job')
    state.update(phase='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
