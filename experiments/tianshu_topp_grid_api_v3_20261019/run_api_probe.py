# SPDX-License-Identifier: Apache-2.0
"""Own the API numerical child and archive after its GPU context exits."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200
    root.mkdir(parents=True);child=None
    state=dict(phase='preparing',complete=False,runner_pid=os.getpid(),official_benchmark=False,adopted=False)
    def save():(root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    def stop(signum,frame):raise TimeoutError(str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
    signal.alarm(min(remaining,2100));save()
    try:
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('probe_api.py')),'--output',str(root/'micro'),'--stop-at-utc',args.stop_at_utc]
        with (root/'probe.log').open('w') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(phase='micro',child_pid=child.pid);save();assert child.wait(timeout=min(remaining-20,1900))==0
        result=json.loads((root/'micro/result.json').read_text());assert result['phase']=='complete' and result['complete']
        state.update(phase='complete',complete=True)
    except BaseException as error:state.update(phase='failed',error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        signal.alarm(0);save()
        (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
    if state['complete']:
        subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'archive_completed_pair.py'),
            '--single-probe','--source',str(root),'--label','topp_grid_api_v3_micro_20261019'],check=True)
    return 0 if state['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
