# SPDX-License-Identifier: Apache-2.0
"""Run independent GPU probes only after the current quality/formal chain exits."""
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
    args=parser.parse_args();root=args.output.resolve();assert not root.exists()
    base=Path(__file__).resolve().parents[1]
    jobs=[('decode_registers',base/'tianshu_decode_registers_20261014/probe_registers.py',['--zero-init'],17),
          ('decode_cache_policy',base/'tianshu_decode_cache_policy_20261014/probe_cache.py',['--zero-init'],17),
          ('prefill_gemm_m4096',base/'tianshu_prefill_gemm_m4096_20261014/probe_mm.py',[],40)]
    sources=[Path(__file__).resolve(),*(j[1] for j in jobs),
             base/'tianshu_decode_low_warps_20261014/launch_router.py',
             base/'tianshu_native_prefill_20261012/zero_initialization.py',
             base/'tianshu_decode_cache_policy_20261014/experimental_cache.py',
             base/'tianshu_prefill_gemm_tiles_20261014/experimental_mm.py']
    upstream=json.loads((args.wait_for/'state.json').read_text())
    assert upstream['runner_pid']==args.upstream_pid
    sources += [Path(p) for p in upstream['frozen_sources']]
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    deadline=datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    remaining=int((deadline-datetime.now(timezone.utc)).total_seconds());assert remaining>7200
    root.mkdir(parents=True)
    for path in frozen:
        p=Path(path);dest=root/'source_snapshot'/p.relative_to(base)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    state=dict(phase='waiting_for_quality_and_official',runner_pid=os.getpid(),
               upstream_root=str(args.wait_for),upstream_pid=args.upstream_pid,
               frozen_sources=frozen,stop_at_utc=deadline.isoformat(),
               official_benchmark=False,adopted=False,completed_jobs=[])
    child=None

    class Stopped(BaseException):pass

    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')

    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise Stopped(str(signum))

    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
    signal.alarm(remaining);save()
    try:
        while True:
            upstream=json.loads((args.wait_for/'state.json').read_text())
            assert upstream['runner_pid']==args.upstream_pid
            proc=Path('/proc')/str(args.upstream_pid)/'stat'
            alive=proc.exists() and proc.read_text().split()[2]!='Z'
            if not alive:
                if upstream['phase']!='complete':
                    state.update(phase='held_for_upstream_review',reason=upstream['phase']);break
                assert Path('/tmp/tianshu_budget4096_buckets83_S8_official4k_20261014_manifest.json').exists()
                state['upstream_completion_verified']=True;break
            time.sleep(10)
        if state.get('upstream_completion_verified'):
            for name,script,extra,count in jobs:
                if (deadline-datetime.now(timezone.utc)).total_seconds()<=3600:
                    state.update(phase='held_for_lease_limit');break
                assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
                assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
                with socket.socket() as connection:
                    assert connection.connect_ex(('127.0.0.1',9031))!=0
                env=dict(os.environ);env.pop('PYTHONPATH',None)
                target=root/name
                with (root/(name+'.log')).open('w') as log:
                    child=subprocess.Popen(['python3','-u',str(script),'--output',str(target),*extra],
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env,cwd='/workspace')
                    state.update(phase='running',current_job=name,child_pid=child.pid);save()
                    assert child.wait(timeout=2800)==0,name
                data=json.loads((target/'result.json').read_text())
                assert data['complete'] and len(data['cases'])==count
                hardware=subprocess.check_output(['ixsmi'],text=True)
                assert 'No running processes found' in hardware
                (target/'hardware_after.txt').write_text(hardware)
                subprocess.run(['python3',str(base/'archive_completed_pair.py'),'--single-probe',
                    '--source',str(target),'--label',name+'_post_quality_20261014'],check=True,
                    stdout=(root/(name+'_archive.log')).open('w'))
                state['completed_jobs'].append(name);save()
            else:state['phase']='complete'
    except Stopped as error:state.update(phase='stopped',error=str(error))
    except Exception as error:state.update(phase='failed',error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' or state['phase'].startswith('held_') else 1


if __name__=='__main__':raise SystemExit(main())
