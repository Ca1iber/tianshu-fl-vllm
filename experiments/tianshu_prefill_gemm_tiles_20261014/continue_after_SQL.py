# SPDX-License-Identifier: Apache-2.0
"""Queue one isolated Prefill tile scan after SQL finishes; preserve contenders."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-for',type=Path,required=True)
    parser.add_argument('--upstream-pid',type=int,required=True)
    args=parser.parse_args()
    root=args.output.resolve()
    assert not root.exists()
    deadline=datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    remaining=int((deadline-datetime.now(timezone.utc)).total_seconds())
    assert remaining>3600
    signal.alarm(remaining)
    root.mkdir(parents=True)
    base=Path(__file__).resolve().parent
    paths=[Path(__file__).resolve(),base/'probe_mm.py',base/'experimental_mm.py',
           base.parent/'tianshu_sql_workflow_20261014/continue_after_MM.py',
           base.parent/'tianshu_sql_workflow_20261014/run_service_pair.py',
           base.parent/'tianshu_sql_workflow_20261014/run_service_trial.py',
           base.parent/'tianshu_sql_workflow_20261014/hooks/sitecustomize.py',
           base.parent/'tianshu_sql_workflow_20261014/reflect_one.py']
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in paths:
        dest=root/'source_snapshot'/p.relative_to(base.parent)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    state=dict(phase='waiting_for_SQL_completion',runner_pid=os.getpid(),upstream_root=str(args.wait_for),
               upstream_pid=args.upstream_pid,frozen_sources=frozen,stop_at_utc=deadline.isoformat(),
               official_benchmark=False,adopted=False)
    child=None

    class Stopped(BaseException):pass

    def save():
        temp=root/'state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'state.json')

    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise Stopped(str(signum))

    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(signum,stop)
    save()
    try:
        while True:
            upstream=json.loads((args.wait_for/'state.json').read_text())
            assert upstream['runner_pid']==args.upstream_pid
            phase=upstream['phase']
            if phase.startswith('held_'):
                state.update(phase='held_for_upstream_review',reason=phase);save();return 0
            if phase in ('failed','stopped'):raise Stopped('Upstream failed/stopped')
            proc=Path('/proc')/str(args.upstream_pid)/'stat'
            alive=proc.exists() and proc.read_text().split()[2]!='Z'
            if phase=='complete' and not alive:break
            time.sleep(5)
        proof=json.loads((args.wait_for/'SQL_reflection_service/comparison.json').read_text())
        assert proof['complete'] and proof['successful_requests']==384 and proof['actual_SQL_mode_verified']
        assert Path('/tmp/tianshu_SQL_reflection_S8_service_20261014_manifest.json').exists()
        state['upstream_completion_verified']=True
        if proof['gain_percent']>=2:
            state.update(phase='held_for_SQL_confirmation',gain_percent=proof['gain_percent']);save();return 0
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
        try:connection=socket.create_connection(('127.0.0.1',9031),timeout=2)
        except OSError:pass
        else:
            connection.close();raise RuntimeError('Port9031 occupied')
        assert (deadline-datetime.now(timezone.utc)).total_seconds()>2700
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in frozen.items())
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        with (root/'prefill_gemm_tiles.log').open('w') as log:
            child=subprocess.Popen(['python3','-u',str(base/'probe_mm.py'),'--output',str(root/'prefill_gemm_tiles')],
                                   stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            state.update(phase='running',child_pid=child.pid,current_job='prefill_gemm_tiles');save()
            assert child.wait(timeout=2700)==0
        result=json.loads((root/'prefill_gemm_tiles/result.json').read_text())
        assert result['complete'] and len(result['cases'])==48
        hardware=subprocess.check_output(['ixsmi'],text=True)
        assert 'No running processes found' in hardware
        (root/'prefill_gemm_tiles/hardware_after.txt').write_text(hardware)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in frozen.items())
        subprocess.run(['python3',str(base.parent/'archive_completed_pair.py'),'--single-probe',
                        '--source',str(root/'prefill_gemm_tiles'),'--label','prefill_gemm_tiles_20261014'],check=True)
        state.update(phase='complete',passing_cases=result['numerically_passing_cases'])
    except Stopped as error:state.update(phase='stopped',error=str(error))
    except Exception as error:state.update(phase='failed',error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
