# SPDX-License-Identifier: Apache-2.0
"""Serialized original-command bounded prefill cadence3+3 framework screen."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stop-at-utc',required=True)
    args=parser.parse_args();root=args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True);child=None
    state=dict(phase='preparing',runner_pid=os.getpid(),official_benchmark=False,adopted=False,
               started_utc=datetime.now(timezone.utc).isoformat(),stop_at_utc=args.stop_at_utc,execution_order=['control','candidate'])

    def save():(root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    def stop(signum,frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save()
    try:
        for mode in ('control','candidate'):
            state['phase']=mode;save()
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            command=[sys.executable,'-u',str(Path(__file__).with_name('run_service_trial.py')),'--output',str(root/mode),
                     '--mode',mode,'--stop-at-utc',args.stop_at_utc]
            with (root/(mode+'.log')).open('w') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['child_pid']=child.pid;save()
                assert child.wait()==0,mode+' failed'
        control=json.loads((root/'control/result.json').read_text());candidate=json.loads((root/'candidate/result.json').read_text())
        assert control['phase']==candidate['phase']=='complete'
        assert control['frozen_sources']==candidate['frozen_sources']
        assert control['server_command']==candidate['server_command']
        a,b=control['profile_environment'],candidate['profile_environment']
        for key in a:
            if key=='TIANSHU_PREFILL_CADENCE_MODE':assert (a[key],b[key])==('control','candidate')
            else:assert a[key].replace(str(root/'control'),'<RUN>')==b[key].replace(str(root/'candidate'),'<RUN>'),key
        assert control['seeded_canaries'][0]['text']==candidate['seeded_canaries'][0]['text']
        concurrent_matches=sum(a['text']==b['text'] for ca,cb in zip(control['concurrent_canaries'],candidate['concurrent_canaries']) for a,b in zip(ca,cb))
        result=dict(complete=True,official_benchmark=False,adopted=False,frozen_sources=control['frozen_sources'],
                    original_serve_command_verified=True,all768_screen_requests_successful=True,seeded_C1_canary_cross_mode_identical=True,
                    concurrent_greedy_cross_mode_equal_count=concurrent_matches,
                    exact_resumed_GPU_input_check_is_correctness_gate=True,
                    control_total_tok_s=control['mean_total_tok_s'],candidate_total_tok_s=candidate['mean_total_tok_s'],
                    gain_percent=100*(candidate['mean_total_tok_s']/control['mean_total_tok_s']-1),
                    requests_per_round=128,rounds_per_mode=3,excluded_rounds=[1],different_input_seeds_each_round=True,
                    RNG_unchanged=True,fresh_Math_not_run=True)
        (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');state['phase']='complete';save()
        subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'archive_completed_pair.py'),
                        '--source',str(root),'--label','prefill_cadence16_service_20261016'],check=True)
    except BaseException as error:
        state.update(phase='failed',error=repr(error));save()
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
        raise


if __name__=='__main__':main()
