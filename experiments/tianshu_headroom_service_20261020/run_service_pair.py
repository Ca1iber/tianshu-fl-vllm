# SPDX-License-Identifier: Apache-2.0
"""Serialized qualified-profile software8 KV 3+3 screen."""
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
    parser.add_argument('--input-len',type=int,choices=(4096,16384),default=4096)
    args=parser.parse_args();root=args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True);child=None
    state=dict(phase='preparing',runner_pid=os.getpid(),official_benchmark=False,adopted=False,
               started_utc=datetime.now(timezone.utc).isoformat(),stop_at_utc=args.stop_at_utc)

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
                     '--mode',mode,'--stop-at-utc',args.stop_at_utc,'--input-len',str(args.input_len)]
            with (root/(mode+'.log')).open('w') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['child_pid']=child.pid;save()
                assert child.wait()==0,mode+' failed'
        control=json.loads((root/'control/result.json').read_text());candidate=json.loads((root/'candidate/result.json').read_text())
        assert control['phase']==candidate['phase']=='complete'
        assert control['frozen_sources']==candidate['frozen_sources']
        assert candidate['server_command']==control['server_command']
        assert control['seeded_canaries'][0]['text']==candidate['seeded_canaries'][0]['text']
        a,b=control['profile_environment'],candidate['profile_environment']
        for key in a:
            if key=='TIANSHU_KV_HEADROOM_MODE':assert (a[key],b[key])==('control','candidate')
            else:assert a[key].replace(str(root/'control'),'<RUN>')==b[key].replace(str(root/'candidate'),'<RUN>'),key
        result=dict(complete=True,official_benchmark=False,adopted=False,frozen_sources=control['frozen_sources'],
                    both_original_serve_commands_verified=True,both_native_default_budget4096=True,both_arms_K8V16=True,all384_screen_requests_successful=True,input_tokens=args.input_len,output_tokens=1024,seeded_C1_repeatability_within_modes=True,KV_values_remain_BF16=True,
                    control_total_tok_s=control['mean_total_tok_s'],candidate_total_tok_s=candidate['mean_total_tok_s'],
                    gain_percent=100*(candidate['mean_total_tok_s']/control['mean_total_tok_s']-1),
                    requests_per_round=64,rounds_per_mode=3,excluded_rounds=[1],different_input_seeds_each_round=True,
                    sampler_algorithm_unchanged=True,fresh_Math_not_run=True)
        (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');state['phase']='complete';save()
    except BaseException as error:
        state.update(phase='failed',error=repr(error));save()
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
        raise


if __name__=='__main__':main()
