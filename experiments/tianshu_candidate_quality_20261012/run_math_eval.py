# SPDX-License-Identifier: Apache-2.0
"""Independent official Level3 evaluation of a process-only Tianshu candidate."""
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
    parser.add_argument('--adaptive',choices=['1'],default='1')
    parser.add_argument('--candidate',choices=['small-gemm','prefill-gemm','sampler','native-prefill','decode-launch'],required=True)
    parser.add_argument('--max-batched-tokens',type=int,choices=[2048,4096,8192],default=2048)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.candidate not in ('native-prefill','decode-launch') and args.max_batched_tokens!=2048:
        parser.error('GEMM/sampler candidates require their independently controlled budget2048 profile')
    if (root/'result.json').exists():raise RuntimeError('Refusing to overwrite a completed evaluation')
    state={'max_batched_tokens':args.max_batched_tokens,'phase':'preparing','candidate':f'hybrid_adaptive_fa2_tile128_budget{args.max_batched_tokens}_{args.candidate}','reused_original_accuracy':False,
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
        base=Path(__file__).resolve().parents[1]
        candidates={'small-gemm':('tianshu_small_gemm_20261012','TIANSHU_SMALL_GEMM_PROBE','TIANSHU_SMALL_GEMM_RECORD_DIR','small_mm_records','mm_*.json'),
                    'prefill-gemm':('tianshu_prefill_gemm_20261012','TIANSHU_PREFILL_GEMM_PROBE','TIANSHU_PREFILL_GEMM_RECORD_DIR','prefill_mm_records','mm_*.json'),
                    'sampler':('tianshu_sampler_fusion_20261012','TIANSHU_SAMPLER_FUSION_PROBE','TIANSHU_SAMPLER_RECORD_DIR','sampler_records','sampler_*.json'),
                    'native-prefill':('tianshu_native_prefill_20261012','TIANSHU_NATIVE_PREFILL_PROBE','TIANSHU_NATIVE_PREFILL_RECORD_DIR','native_prefill_records','native_prefill_*.json'),
                    'decode-launch':('tianshu_decode_launch_20261012','TIANSHU_DECODE_LAUNCH_PROBE','TIANSHU_DECODE_LAUNCH_RECORD_DIR','decode_launch_records','launch_*.json')}
        folder,flag,record_flag,record_dir,pattern=candidates[args.candidate]
        for _, candidate_flag, candidate_record_flag, _, _ in candidates.values():
            env.pop(candidate_flag, None)
            env.pop(candidate_record_flag, None)
        hook=base/folder/'hooks/sitecustomize.py'
        state['experimental_candidate']=args.candidate
        env[flag]='1';env[record_flag]=str(root/record_dir)
        if args.candidate=='decode-launch':
            env['TIANSHU_NATIVE_PREFILL_PROBE']='1'
            env['TIANSHU_NATIVE_PREFILL_RECORD_DIR']=str(root/'native_prefill_records')
        env['PYTHONPATH']=str(hook.parent)
        env['TIANSHU_FA2_TILE_PROBE']='1'
        env['TIANSHU_FA2_TILE_RECORD_DIR']=str(root/'tile_records')
        state['tile_hook']={'path':str(hook),'sha256':hashlib.sha256(hook.read_bytes()).hexdigest()}
        state['tile_configuration']={'BLOCK_M':128,'BLOCK_N':128,'num_warps':8,'num_stages':2}
        command=['vllm','serve','/workspace/MiniCPM5-2B','--port','9031',
            '--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}',
            '--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
        command += ['--max-num-batched-tokens',str(args.max_batched_tokens)]
        state['server_command']=command
        source=Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker/iluvatar_hybrid_attention.py')
        state['hybrid_source']=str(source);state['hybrid_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        adaptive=source.with_name('iluvatar_adaptive_decode.py');state['adaptive_flag']=args.adaptive;state['adaptive_source']=str(adaptive);state['adaptive_sha256']=hashlib.sha256(adaptive.read_bytes()).hexdigest()
        snapshots=root/'source_snapshot';snapshots.mkdir(exist_ok=True)
        (snapshots/'candidate_sitecustomize.py').write_bytes(hook.read_bytes())
        tile_hook=base/'tianshu_fa2_tiles_20261012/hooks/sitecustomize.py'
        (snapshots/'tile_sitecustomize.py').write_bytes(tile_hook.read_bytes())
        state['candidate_hook']=dict(state['tile_hook'])
        state['tile_hook']={'path':str(tile_hook),'sha256':hashlib.sha256(tile_hook.read_bytes()).hexdigest()}
        kernel=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py') if args.candidate=='decode-launch' else base/('tianshu_sampler_fusion_20261012/experimental_sampler.py' if args.candidate=='sampler' else
                    'tianshu_native_prefill_20261012/zero_initialization.py' if args.candidate=='native-prefill' else
                    'tianshu_splitk_20261012/experimental_mm.py')
        (snapshots/kernel.name).write_bytes(kernel.read_bytes())
        state['experimental_kernel']={'path':str(kernel),'sha256':hashlib.sha256(kernel.read_bytes()).hexdigest()}
        if args.candidate=='decode-launch':
            native_hook=base/'tianshu_native_prefill_20261012/hooks/sitecustomize.py'
            generator=base/'tianshu_native_prefill_20261012/zero_initialization.py'
            (snapshots/'native_prefill_sitecustomize.py').write_bytes(native_hook.read_bytes())
            (snapshots/'zero_initialization.py').write_bytes(generator.read_bytes())
            state['native_prefill_hook']={'path':str(native_hook),'sha256':hashlib.sha256(native_hook.read_bytes()).hexdigest()}
            state['zero_generator']={'path':str(generator),'sha256':hashlib.sha256(generator.read_bytes()).hexdigest()}
        for name in ('worker.py','iluvatar_hybrid_attention.py','iluvatar_adaptive_decode.py'):
            current=source.with_name(name);contents=current.read_bytes();(snapshots/name).write_bytes(contents)
            if name=='worker.py':state['worker_sha256']=hashlib.sha256(contents).hexdigest()
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
        records=list((root/'tile_records').glob('tile_*.json'))
        if not records and args.candidate not in ('native-prefill','decode-launch'):raise RuntimeError('Actual model did not use the experimental tile')
        state['tile_routing_records']=[json.loads(p.read_text()) for p in records]
        routes=[json.loads(p.read_text()) for p in (root/record_dir).glob(pattern)]
        if not routes:raise RuntimeError('Experimental candidate did not route in the evaluated model')
        if args.candidate in ('small-gemm','prefill-gemm') and len({tuple(x['shape']) for x in routes})!=3:
            raise RuntimeError('Evaluated GEMM candidate did not route all three checked shapes')
        state['candidate_routing_records']=routes
        if args.candidate=='decode-launch':
            native_routes=[json.loads(p.read_text()) for p in (root/'native_prefill_records').glob('native_prefill_*.json')]
            assert native_routes and all(x['kernel_sha256']==state['zero_generator']['sha256'] for x in native_routes)
            assert all(x['configuration']=={'TILE_SIZE':16,'num_warps':4,'num_stages':1} for x in routes)
            state['native_prefill_routing_records']=native_routes
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
