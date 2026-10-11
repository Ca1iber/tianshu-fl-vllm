# SPDX-License-Identifier: Apache-2.0
"""Serialize the corrected CPU oracle and compile-verified GEMM full scan."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    source=Path(__file__).resolve().parents[1]
    root=Path('/workspace/benchmark-results/20261012_next_diagnostics_tianshu')
    state_file=root/'corrected_state.json'
    assert not state_file.exists(), 'Refusing duplicate corrected probe controller'
    smoke=json.loads((root/'splitk_v2_smoke/result.json').read_text())
    assert smoke['complete'] and len(smoke['cases'])==3
    assert all(c.get('vs_fp32',{}).get('passed') and c.get('vs_cpu',{}).get('passed') and 'error' not in c for c in smoke['cases'])
    state={'phase':'preparing','jobs':[],'started_utc':datetime.now(timezone.utc).isoformat()}
    def save():
        state_file.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    commands=[
        [str(source/'tianshu_sampler_fusion_20261012/probe_sampler.py'),'--output',str(root/'sampler_v2_cpu')],
        [str(source/'tianshu_splitk_20261012/probe_mm.py'),'--output',str(root/'splitk_v2_full')],
    ]
    for index,args in enumerate(commands):
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
        state.update(phase=f'running_{index}',active_command=[sys.executable,'-u',*args]);save()
        with (root/f'corrected_job_{index}.log').open('w') as log:
            proc=subprocess.Popen(state['active_command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['active_pid']=proc.pid;save();code=proc.wait()
        state['jobs'].append({'command':state['active_command'],'exit_code':code,
                              'script_sha256':hashlib.sha256(Path(args[0]).read_bytes()).hexdigest()})
        save()
    state.update(phase='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()


if __name__=='__main__':
    main()
