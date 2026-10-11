# SPDX-License-Identifier: Apache-2.0
"""Short verified-hybrid service profiles; diagnostic timings are not benchmark scores."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import threading
import time
import traceback
import urllib.request


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-graphs',action='store_true')
    parser.add_argument('--small-only',action='store_true')
    parser.add_argument('--adaptive',choices=['0','1'],default='0')
    parser.add_argument('--max-batched-tokens',type=int,choices=[2048,4096,8192],default=2048)
    parser.add_argument('--warmup-rounds',type=int,default=2)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.warmup_rounds < 1:parser.error('At least one full-shape warmup round is required')
    if (root/'result.json').exists():raise RuntimeError('Refusing to overwrite a completed profile')
    traces=root/'traces'
    profiler={'profiler':'torch','torch_profiler_dir':str(traces),
        'torch_profiler_with_stack':False,'torch_profiler_record_shapes':False,
        'torch_profiler_with_memory':False,'ignore_frontend':True,
        'torch_profiler_dump_cuda_time_total':True,'torch_profiler_use_gzip':True}
    command=['vllm','serve','/workspace/MiniCPM5-2B','--port','9031',
        '--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}',
        '--served-model-name','minicpm','--gpu-memory-utilization','0.85',
        '--max-model-len','131072','--profiler-config',json.dumps(profiler)]
    command += ['--max-num-batched-tokens', str(args.max_batched_tokens)]
    if args.no_graphs:
        command[command.index('--compilation-config')+1]='{"cudagraph_mode":"NONE"}'
    state={'phase':'preparing','official_benchmark':False,'command':command,
           'runner_pid':os.getpid(),'cases':[],'started_utc':datetime.now(timezone.utc).isoformat(),
           'diagnostic_graphs_disabled':args.no_graphs,
           'adaptive':args.adaptive, 'max_batched_tokens':args.max_batched_tokens}
    server=None
    deadline=datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])

    class StopRequested(BaseException): pass
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if server is not None and server.poll() is None: os.killpg(server.pid,signal.SIGTERM)
        raise StopRequested(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def watchdog():
        while datetime.now(timezone.utc)<deadline: time.sleep(5)
        os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=watchdog,daemon=True).start()
    def save():
        p=root/'state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
        print(json.dumps({'phase':state['phase'],'cases':len(state['cases']),'error':state.get('error')}),flush=True)
    def post(endpoint,body,timeout=900):
        request=urllib.request.Request('http://127.0.0.1:9031'+endpoint,data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(request,timeout=timeout) as response: return response.read()
    def completion(output,seed):
        rng=random.Random(seed)
        body={'model':'minicpm','prompt':[rng.randrange(512,120000) for _ in range(4096)],
            'max_tokens':output,'temperature':1.,'top_p':.95,'ignore_eos':True,'stream':False}
        start=time.perf_counter();response=json.loads(post('/v1/completions',body))
        if response['usage']['prompt_tokens']!=4096 or response['usage']['completion_tokens']!=output:
            raise RuntimeError('Wrong usage: '+str(response.get('usage')))
        return {'wall_seconds':time.perf_counter()-start,'usage':response['usage']}
    def batch(concurrency,output,seed):
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures=[pool.submit(completion,output,seed+i) for i in range(concurrency)]
            results=[f.result() for f in futures]
        return {'wall_seconds':time.perf_counter()-start,'responses':results}
    def analyze(paths):
        total,calls,classes=Counter(),Counter(),Counter()
        for path in paths:
            if not path.name.endswith('.json.gz'):continue
            with gzip.open(path,'rt') as stream:data=json.load(stream)
            for e in data.get('traceEvents',[]):
                if e.get('cat','').lower()=='kernel' and e.get('ph')=='X' and e.get('dur',0)>0:
                    name=e['name'];total[name]+=e['dur'];calls[name]+=1
                    grid=e.get('args',{}).get('grid',[])
                    if name=='kernel_unified_attention':
                        if isinstance(grid,list) and len(grid)==3:
                            classes['attention_3d' if grid[2]>1 else 'attention_2d']+=e['dur']
                        else:classes['attention_grid_unknown']+=e['dur']
        summed=sum(total.values())
        return {'gpu_kernel_total_ms':summed/1000,
            'attention_classes_ms':{k:v/1000 for k,v in classes.items()},
            'top_kernels':[{'name':name,'calls':calls[name],'total_ms':value/1000,
                'percent':100*value/summed} for name,value in total.most_common(30)]}
    try:
        try:s=socket.create_connection(('127.0.0.1',9031),timeout=2)
        except OSError:pass
        else:s.close();raise RuntimeError('Port9031 occupied')
        env=dict(os.environ,VLLM_PLUGINS='fl',VLLM_FL_USE_FLAGGEMS_ATTN='0',
            FLAGGEMS_ILUVATAR_COMPACT_VARLEN='1',
            VLLM_FL_ILUVATAR_HYBRID_ATTENTION='1',
            VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS=args.adaptive,VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN='0',
            VLLM_FL_ILUVATAR_PREFILL_BUCKETS='0',VLLM_FL_ILUVATAR_LOGITS_GATHER='0',
            TRITON_PRINT_AUTOTUNING='1')
        env.pop('TIANSHU_PACK_PROBE_MODE',None);env.pop('PYTHONPATH',None)
        hardware=subprocess.check_output(['ixsmi'],text=True)
        (root/'hardware_before.txt').write_text(hardware)
        if 'No running processes found' not in hardware:raise RuntimeError('GPU not idle')
        installed=Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker')
        snapshots=root/'source_snapshot';snapshots.mkdir(exist_ok=True)
        state['installed_sources']={}
        for name in ('worker.py','iluvatar_hybrid_attention.py','iluvatar_adaptive_decode.py'):
            source=installed/name;contents=source.read_bytes();(snapshots/name).write_bytes(contents)
            state['installed_sources'][name]={'path':str(source),'sha256':hashlib.sha256(contents).hexdigest()}
        with (root/'server.log').open('w') as log:
            server=subprocess.Popen(command,cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='starting',server_pid=server.pid);save()
        limit=time.monotonic()+2400
        while True:
            if server.poll() is not None:raise RuntimeError('Server exited')
            if time.monotonic()>limit:raise TimeoutError('Startup >40min')
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response:d=json.load(response)
                if any(x['id']=='minicpm' for x in d['data']):break
            except Exception:pass
            time.sleep(3)
        for index,concurrency in enumerate((16,) if args.small_only else (16,64)):
            state['phase']=f'warmup_{concurrency}';save()
            warmup=[batch(concurrency,128,2026101140+1000*index+100*round_index)
                    for round_index in range(args.warmup_rounds)]
            before=set(traces.rglob('*'))
            log_boundary=len((root/'server.log').read_text(errors='replace'))
            state['phase']=f'profiling_{concurrency}';save()
            post('/start_profile',{})
            try:measured=batch(concurrency,128,2026101540+1000*index)
            finally:post('/stop_profile',{},timeout=300)
            paths=[p for p in traces.rglob('*') if p.is_file() and p not in before]
            analysis=analyze(paths)
            if analysis['gpu_kernel_total_ms']<=0:raise RuntimeError('No GPU activity events')
            state['cases'].append({'concurrency':concurrency,'input_tokens':4096,'output_tokens':128,
                'warmup':warmup,'measured':measured,'trace_files':[str(p) for p in paths],'analysis':analysis})
            state['cases'][-1]['observed_tuning_lines']=[
                line for line in (root/'server.log').read_text(errors='replace')[log_boundary:].splitlines()
                if 'autotuning' in line.lower() or 'AABS' in line]
            save()
        state['phase']='complete'
    except BaseException as error:
        state.update(phase='failed',error=str(error),traceback=traceback.format_exc());print(state['traceback'],flush=True)
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
        state['finished_utc']=datetime.now(timezone.utc).isoformat();save()
        (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
    return 0 if state['phase']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
