# SPDX-License-Identifier: Apache-2.0
"""Verify IXsys process-tree scope, Graph/eager event coverage and timing units."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import statistics
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True)
    args=parser.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True);remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>600
    state=dict(phase='preparing',complete=False,runner_pid=os.getpid(),official_benchmark=False,adopted=False,coverage_verified=False)
    child=None
    def save():(root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    def stop(signum,frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise KeyboardInterrupt(str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
    signal.alarm(min(remaining,600));save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_before.txt').write_text(hardware);assert 'No running processes found' in hardware
        base=Path(__file__).resolve().parent
        state['sources']={}
        for path in (Path(__file__),base/'probe_process_tree_v2.py'):
            data=path.read_bytes();state['sources'][str(path)]=hashlib.sha256(data).hexdigest();(root/path.name).write_bytes(data)
        profiler='/usr/local/corex/bin/ixsys'
        for flag,name in (('--help','ixsys_help.txt'),('--version','ixsys_version.txt')):
            (root/name).write_text(subprocess.check_output([profiler,flag],text=True))
        command=[profiler,'-t','cuda','--cuda-trace-scope','process-tree','--profile-from-start','off','-o',str(root/'trace'),
                 '--print-gpu-summary',sys.executable,'-u',str(base/'probe_process_tree_v2.py'),'--output',str(root/'workload')]
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        with (root/'ixsys.log').open('w') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(phase='profiling',profiler_pid=child.pid,command=command);save();code=child.wait(timeout=480)
        state['profiler_exit_code']=code;assert code==0
        workload=json.loads((root/'workload/result.json').read_text());assert workload['complete']
        measured=workload['child'];pids={measured['child_pid'],*measured['namespace_pids']}
        db=sqlite3.connect('file:'+str(root/'trace')+'?mode=ro',uri=True)
        graph_rows=db.execute('select processId,ts,dur from cuda_graph_data order by ts').fetchall()
        kernels=db.execute('select f.name,count(*),sum(k.dur) from cuda_kernel_data k join function_name_map f on f.id=k.name_id group by f.name').fetchall()
        kernel_pids={p for p, in db.execute('select distinct processId from cuda_kernel_data')}
        outside=sum(n for name,n,dur in kernels if 'tianshu_stage_outside' in name)
        inside={short:sum(n for name,n,dur in kernels if short in name) for short in measured['expected_inside_calls']}
        api_calls=db.execute('select count(*) from cuda_api_data').fetchone()[0]
        assert len(graph_rows)==measured['expected_graph_calls'] and all(pid in pids and dur>0 for pid,ts,dur in graph_rows),graph_rows
        assert kernel_pids and kernel_pids<=pids and outside==20,(kernel_pids,kernels)
        graph_trace_ms=sum(dur for pid,ts,dur in graph_rows)/1e6
        trace_postwarm_median=statistics.median(row[2]/1e6 for row in graph_rows[measured['profiled_graph_warmup_count']:])
        event_unprofiled_median=statistics.median(measured['unprofiled_single_before_ms']+measured['unprofiled_single_after_ms'])
        ratio=event_unprofiled_median/trace_postwarm_median
        assert .5<ratio<1.5,(event_unprofiled_median,trace_postwarm_median)
        audit=dict(complete=True,process_tree_child_coverage_verified=True,graph_calls=len(graph_rows),outside_kernel_calls=outside,
                   expected_inside_kernel_calls=measured['expected_inside_calls'],observed_inside_kernel_calls=inside,
                   internal_graph_kernel_coverage_verified=inside==measured['expected_inside_calls'],
                   cuda_api_calls=api_calls,graph_trace_sum_ms=graph_trace_ms,trace_graph_postwarm_median_ms=trace_postwarm_median,
                   unprofiled_event_graph_median_ms=event_unprofiled_median,unprofiled_events_to_trace_postwarm_ratio=ratio,
                   profiled_event_graph_median_ms=statistics.median(measured['graph_events_ms']),
                   comparison_scope='Unprofiled outside-event median vs postwarm trace Graph median; profiled intervals retain instrumentation gaps',
                   trace_duration_units='ns; cross-checked against unprofiled outside-Graph events',
                   child_pid_candidates=sorted(pids),kernel_names_counts_durations_ns=kernels,
                   graph_rows=graph_rows,trace_sha256=hashlib.sha256((root/'trace').read_bytes()).hexdigest())
        (root/'coverage_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        state.update(phase='complete',complete=True,coverage_verified=True,audit=audit,workload=workload)
    except BaseException as error:
        state.update(phase='failed',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        signal.alarm(0);(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True));save()
        (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    if state['phase']=='complete':subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'archive_completed_pair.py'),
        '--single-probe','--source',str(root),'--label','graph_stage_process_tree_v2_20261016'],check=True)


if __name__=='__main__':main()
