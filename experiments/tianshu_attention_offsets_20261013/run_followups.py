# SPDX-License-Identifier: Apache-2.0
"""Serialize authorized follow-up validation with conditional official reruns."""
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
    state=dict(phase='waiting_for_address_service',runner_pid=os.getpid(),jobs=[],adopted=False,
               canonical_best_modified=False,service_root=str(args.service_root),stop_at_utc=deadline.isoformat())
    (root/'run_followups.py').write_bytes(Path(__file__).read_bytes())

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
        state['address_service_gain_percent']=comparison['gain_percent'];save()
        # Each job starts only after the prior server/runner has fully exited.
        run('kv_micro',['python3','-u',str(base/'tianshu_kv_offsets_20261013/probe_kv.py'),'--output',str(root/'kv_micro')])
        micro=json.loads((root/'kv_micro/result.json').read_text());assert micro['complete'] and micro['numerical_checks_passed']
        archive('kv_micro','kv_offsets_width_launch_micro_20261013')
        ops=Path('/workspace/FlagGems/experiments/tianshu_sql_reflection_20261013')
        run('sql_cpu',['python3','-u',str(ops/'probe_cpu.py'),'--output',str(root/'sql_cpu')])
        cpu=json.loads((root/'sql_cpu/result.json').read_text());assert cpu['complete'] and cpu['all_checks_passed']
        snapshot=root/'sql_cpu/source_snapshot';snapshot.mkdir()
        for p in [ops/'probe_cpu.py',ops/'reflect_one.py']:(snapshot/p.name).write_bytes(p.read_bytes())
        (snapshot/'ops_git_head.txt').write_text(subprocess.check_output(['git','-C','/workspace/FlagGems','rev-parse','HEAD'],text=True))
        archive('sql_cpu','sql_target_reflection_cpu_remote_20261013')
        run('sql_service',['python3','-u',str(ops/'run_service.py'),'--output',str(root/'sql_service')])
        archive('sql_service','sql_target_reflection_service_20261013')
        clear=(comparison['gain_percent']>=5 and min(comparison['candidate_rounds'][1:])>max(comparison['control_rounds'][1:]))
        state['address_improvement_clear']=clear;save()
        if clear:
            run('math',['python3','-u',str(Path(__file__).with_name('run_math.py')),'--output',str(root/'math'),
                        '--service-root',str(args.service_root)])
            quality=json.loads((root/'math/accuracy_audit.json').read_text())
            state.update(math_correct=quality['correct'],math_total=105,math_accuracy=quality['accuracy'],quality_passed=quality['official_requirement_passed']);save()
            if quality['official_requirement_passed']:
                # Run original workloads and four rounds; no benchmark edits.
                for tokens in [16384,4096]:
                    needed=5400 if tokens==16384 else 3600
                    assert (deadline-datetime.now(timezone.utc)).total_seconds()>needed
                    run(f'official{tokens}',['python3','-u',str(Path(__file__).with_name('run_official.py')),
                        '--output',str(root/f'official{tokens}'),'--service-root',str(args.service_root),
                        '--math-results',str(root/'math'),'--input-tokens',str(tokens)])
            else:state['official_skipped_reason']='Independent fresh accuracy did not exceed95%'
        else:state['official_skipped_reason']='Short service gain not clearly above5% with non-overlapping warm rounds'
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
