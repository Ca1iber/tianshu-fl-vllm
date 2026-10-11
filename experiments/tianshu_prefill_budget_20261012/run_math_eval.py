# SPDX-License-Identifier: Apache-2.0
"""Independent official Level3 evaluation for a selected Prefill-budget candidate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback
import urllib.request


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--adaptive',choices=['0','1'],default='1')
    parser.add_argument('--max-batched-tokens',type=int,choices=[2048,4096,8192],required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'result.json').exists():raise RuntimeError('Refusing to overwrite a completed evaluation')
    state={'max_batched_tokens':args.max_batched_tokens,'phase':'preparing','candidate':'hybrid_adaptive_prefill_budget','reused_original_accuracy':False,
           'started_utc':datetime.now(timezone.utc).isoformat(),'runner_pid':os.getpid()}
    server=evaluator=None
    stop_at=datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    class StopRequested(BaseException):pass
    def terminate(process):
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        # End requests before Python unwinds a waiting evaluator.
        if server is not None and server.poll() is None:os.killpg(server.pid,signal.SIGTERM)
        if evaluator is not None and evaluator.poll() is None:os.killpg(evaluator.pid,signal.SIGTERM)
        raise StopRequested(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def watchdog():
        while datetime.now(timezone.utc)<stop_at:time.sleep(5)
        os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=watchdog,daemon=True).start()
    def save():
        temp=root/'state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'state.json')
        print(json.dumps({'phase':state['phase'],'error':state.get('error')}),flush=True)
    try:
        if datetime.now(timezone.utc)>=stop_at:raise StopRequested('Lease ended')
        try:connection=socket.create_connection(('127.0.0.1',9031),timeout=2)
        except OSError:pass
        else:connection.close();raise RuntimeError('Port9031 occupied')
        hardware=subprocess.check_output(['ixsmi'],text=True)
        (root/'hardware_before.txt').write_text(hardware)
        if 'No running processes found' not in hardware:raise RuntimeError('GPU not idle')
        env=dict(os.environ,VLLM_PLUGINS='fl',VLLM_FL_USE_FLAGGEMS_ATTN='0',
            FLAGGEMS_ILUVATAR_COMPACT_VARLEN='1',VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN='0',
            VLLM_FL_ILUVATAR_PREFILL_BUCKETS='0',VLLM_FL_ILUVATAR_LOGITS_GATHER='0',
            VLLM_FL_ILUVATAR_HYBRID_ATTENTION='1',VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS=args.adaptive,VLLM_FL_ILUVATAR_HYBRID_STATS_PATH=str(root/'routing_stats.json'))
        env.pop('PYTHONPATH',None);env.pop('TIANSHU_PACK_PROBE_MODE',None)
        command=['vllm','serve','/workspace/MiniCPM5-2B','--port','9031',
            '--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}',
            '--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
        command += ['--max-num-batched-tokens',str(args.max_batched_tokens)]
        state['server_command']=command
        source=Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker/iluvatar_hybrid_attention.py')
        state['hybrid_source']=str(source);state['hybrid_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        adaptive=source.with_name('iluvatar_adaptive_decode.py');state['adaptive_flag']=args.adaptive;state['adaptive_source']=str(adaptive);state['adaptive_sha256']=hashlib.sha256(adaptive.read_bytes()).hexdigest()
        with (root/'server.log').open('w') as log:
            server=subprocess.Popen(command,cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='starting',server_pid=server.pid);save();limit=time.monotonic()+2400
        while True:
            if server.poll() is not None:raise RuntimeError('Server exited')
            if time.monotonic()>limit:raise TimeoutError('Startup >40min')
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response:d=json.load(response)
                if any(x['id']=='minicpm' for x in d['data']):break
            except Exception:pass
            time.sleep(3)
        evaluation=['evalscope','eval','--model','minicpm','--api-url','http://127.0.0.1:9031/v1/chat/completions',
            '--api-key','EMPTY','--eval-type','openai_api','--datasets','math_500',
            '--dataset-args','{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}',
            '--eval-batch-size','8','--timeout','3600',
            '--generation-config','{"temperature":1.0,"top_p":0.95,"max_tokens":32768}',
            '--work-dir',str(root/'evalscope'),'--ignore-errors']
        state['evaluation_command']=evaluation
        with (root/'evalscope.log').open('w') as log:
            evaluator=subprocess.Popen(evaluation,cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='evaluating',evaluator_pid=evaluator.pid);save()
        code=evaluator.wait();state['evalscope_exit_code']=code
        if code!=0:raise RuntimeError('EvalScope failed: '+str(code))
        state['phase']='evaluation_finished_pending_report_audit'
    except StopRequested as error:state.update(phase='stopped',error=str(error))
    except BaseException as error:state.update(phase='failed',error=str(error),traceback=traceback.format_exc())
    finally:
        terminate(evaluator);terminate(server)
        state['finished_utc']=datetime.now(timezone.utc).isoformat();save()
        (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
    return 0 if state['phase']=='evaluation_finished_pending_report_audit' else 1


if __name__=='__main__':raise SystemExit(main())
