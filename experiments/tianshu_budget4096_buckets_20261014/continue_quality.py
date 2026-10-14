# SPDX-License-Identifier: Apache-2.0
"""Lease-bounded serial confirmation→fresh Math105→original official4K."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    for name in ('output','confirm-root','math-root','official-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not any(p.exists() for p in (root,args.math_root,args.official_root))
    base = Path(__file__).resolve().parent
    confirm_state = json.loads((args.confirm_root/'state.json').read_text())
    paths = [Path(__file__), *(base/name for name in ('run_math_eval.py','audit_math_eval.py',
        'audit_profile.py','run_official_4k.py','audit_official.py','verify_downloaded_confirm.py'))]
    paths += [Path(p) for p in confirm_state['frozen_sources']]
    frozen = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    deadline = datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    remaining = int((deadline-datetime.now(timezone.utc)).total_seconds())
    assert remaining > 3600
    root.mkdir(parents=True)
    for path in frozen:
        p = Path(path);dest = root/'source_snapshot'/p.relative_to(base.parent)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    state = dict(phase='waiting_for_confirm',runner_pid=os.getpid(),stop_at_utc=deadline.isoformat(),
                 frozen_sources=frozen,adopted=False,confirm_root=str(args.confirm_root))
    child = None

    class StopRequested(BaseException):
        pass

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = root/'state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'state.json')

    def stop(signum, frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        raise StopRequested(str(signum))

    def run(phase, command):
        nonlocal child
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
        assert datetime.now(timezone.utc)<deadline
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
        with (root/(phase+'.log')).open('w') as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/workspace')
            state.update(phase=phase,child_pid=child.pid,command=command);save()
            assert child.wait()==0, phase

    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
    signal.alarm(remaining);save()
    try:
        while True:
            upstream = json.loads((args.confirm_root/'state.json').read_text())
            assert upstream['runner_pid']==confirm_state['runner_pid']
            proc = Path('/proc')/str(upstream['runner_pid'])/'stat'
            alive = proc.exists() and proc.read_text().split()[2]!='Z'
            if upstream['phase'] in ('failed','stopped'):raise RuntimeError('Confirmation failed/stopped')
            if not alive:
                assert upstream['phase']=='complete'
                assert Path('/tmp/tianshu_budget4096_buckets83_S8_confirm5_20261014_manifest.json').exists()
                break
            time.sleep(10)
        proof = json.loads((args.confirm_root/'comparison.json').read_text())
        assert proof['complete'] and proof['excluded_rounds']==[1,2] and proof['successful_requests']==640
        if proof['gain_percent']<2 or min(proof['candidate_rounds'][2:])<=max(proof['control_rounds'][2:]):
            state.update(phase='confirmation_rejected',gain_percent=proof['gain_percent'])
        else:
            subprocess.run(['python3',str(base/'verify_downloaded_confirm.py'),'--root',str(args.confirm_root)],check=True,stdout=(root/'confirm_readonly_audit.json').open('w'))
            run('running_math',['python3','-u',str(base/'run_math_eval.py'),'--output',str(args.math_root),
                '--service-root',str(args.confirm_root),'--candidate','decode-coherent','--max-batched-tokens','4096',
                '--buckets','1','--gather','1','--bucket-trace','2'])
            quality = json.loads((args.math_root/'accuracy_audit.json').read_text())
            assert quality['complete'] and quality['total']==105 and not quality['reused_original_accuracy']
            state.update(math_correct=quality['correct'],math_total=105)
            if not quality['official_requirement_passed'] or quality['correct']/105<=.95:
                state['phase']='quality_rejected'
            else:
                run('running_official',['python3','-u',str(base/'run_official_4k.py'),'--output',str(args.official_root),
                    '--control-trial',str(args.confirm_root/'control'),'--candidate-trial',str(args.confirm_root/'candidate'),
                    '--math-results',str(args.math_root),'--max-batched-tokens','4096','--decode-launch','1',
                    '--buckets','1','--gather','1','--bucket-trace','2'])
                state.update(phase='complete',official_root=str(args.official_root),local_download_audit_still_required=True)
    except StopRequested as error:
        state.update(phase='stopped',error=str(error))
    except Exception as error:
        state.update(phase='failed',error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase'] in ('complete','confirmation_rejected','quality_rejected') else 1


if __name__=='__main__':
    raise SystemExit(main())
