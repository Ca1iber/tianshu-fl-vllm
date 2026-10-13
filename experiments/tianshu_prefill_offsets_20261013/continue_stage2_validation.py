# SPDX-License-Identifier: Apache-2.0
"""Run fresh stage2 quality and original workloads only after normal16K gain."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--service-root',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    base=Path(__file__).resolve().parents[1]
    deadline=datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    remaining=int((deadline-datetime.now(timezone.utc)).total_seconds());assert remaining>7200
    signal.alarm(remaining);child=None
    state=dict(phase='waiting_for_normal16k_service',runner_pid=os.getpid(),jobs=[],adopted=False,
               canonical_best_modified=False,service_root=str(args.service_root),stop_at_utc=deadline.isoformat())
    (root/'continue_stage2_validation.py').write_bytes(Path(__file__).read_bytes())

    class StopRequested(BaseException):pass
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise StopRequested(f'Signal {signum}')
    for signum in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(signum,stop)
    def save():
        p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')
    def run(name,command):
        nonlocal child
        assert datetime.now(timezone.utc)<deadline
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
        state.update(phase='running',current_job=name);save()
        with (root/f'{name}_runner.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=child.pid;save();code=child.wait()
        state['jobs'].append(dict(name=name,exit_code=code));save();assert code==0,name
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    def archive(name,label):
        subprocess.run(['python3',str(base/'archive_completed_pair.py'),'--single-probe',
                        '--source',str(root/name),'--label',label],check=True,capture_output=True)
    save()
    try:
        while not (args.service_root/'result.json').exists():
            assert datetime.now(timezone.utc)<deadline;time.sleep(5)
        assert json.loads((args.service_root/'result.json').read_text())['phase']=='complete'
        comparison=json.loads((args.service_root/'comparison.json').read_text())
        state['stage2_normal16k_service_gain_percent']=comparison['gain_percent'];save()
        clear=(comparison['gain_percent']>=2 and min(comparison['candidate_rounds'][1:])>max(comparison['control_rounds'][1:]))
        state['stage2_improvement_clear']=clear;save()
        if clear:
            run('math',['python3','-u',str(Path(__file__).with_name('run_stage2_math.py')),'--output',str(root/'math'),
                        '--service-root',str(args.service_root)])
            quality=json.loads((root/'math/accuracy_audit.json').read_text())
            state.update(math_correct=quality['correct'],math_total=105,math_accuracy=quality['accuracy'],quality_passed=quality['official_requirement_passed']);save()
            if quality['official_requirement_passed']:
                # Run original workloads and four rounds; no benchmark edits.
                for tokens in [16384,4096]:
                    needed=5400 if tokens==16384 else 3600
                    assert (deadline-datetime.now(timezone.utc)).total_seconds()>needed
                    run(f'official{tokens}',['python3','-u',str(Path(__file__).with_name('run_stage2_official.py')),
                        '--output',str(root/f'official{tokens}'),'--service-root',str(args.service_root),
                        '--math-results',str(root/'math'),'--input-tokens',str(tokens)])
            else:state['official_skipped_reason']='Independent fresh accuracy did not exceed95%'
        else:state['official_skipped_reason']='Normal16K service gain not clearly above2% with non-overlapping warm rounds'
        state['phase']='complete'
    except StopRequested as error:state.update(phase='stopped',error=str(error))
    except Exception as error:state.update(phase='failed',error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
