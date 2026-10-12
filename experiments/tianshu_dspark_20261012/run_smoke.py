# SPDX-License-Identifier: Apache-2.0
"""Bring up the DSpark compatibility lane, retain every response and clean up."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import random
import statistics
import subprocess
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--diagnostic-prefixes',action='store_true')
    parser.add_argument('--zero-verify',action='store_true')
    parser.add_argument('--quick-performance',action='store_true')
    parser.add_argument('--concurrency',type=int,choices=[8,64],default=8)
    parser.add_argument('--target-graphs',action='store_true')
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    assert not (root/'state.json').exists()
    lease=Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    deadline=datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
    remaining=int((deadline-datetime.now(timezone.utc)).total_seconds())
    assert remaining>0,'Reservation has ended'
    signal.alarm(min(3600,remaining))
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    directory=Path(__file__).resolve().parent
    draft=Path('/workspace/MiniCPM5-2B-DSpark')
    cfg=json.loads((draft/'config.json').read_text())
    # The published auto_map names an SGLang/Transformers helper not shipped
    # in this checkpoint. Use our registered native vLLM draft class.
    cfg.pop('auto_map',None)
    assert cfg['target_layer_ids']==[1,10,20,30,39] and cfg['block_size']==7
    alias=root/'draft_config_alias';alias.mkdir()
    cfg['dflash_config']={'target_layer_ids':cfg['target_layer_ids'],'mask_token_id':cfg['mask_token_id'],
                          'block_size':7,'projector_type':'dspark','use_aux_hidden_state':True,'causal':False}
    (alias/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (alias/'model.safetensors').symlink_to(draft/'model.safetensors')
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for p in directory.rglob('*.py'):
        if 'results' in p.parts:continue
        target=snapshot/p.relative_to(directory);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    installed=Path('/usr/local/lib/python3.12/site-packages')
    sources={}
    for rel in ['vllm_fl/worker/worker.py','vllm_fl/worker/model_runner.py',
                'vllm/config/speculative.py','vllm/v1/spec_decode/dflash.py',
                'vllm/v1/spec_decode/llm_base_proposer.py','vllm/model_executor/models/qwen3_dflash.py']:
        p=installed/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
        target=snapshot/'installed'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    state={'phase':'preparing','official_benchmark':False,'accuracy_test':False,
           'speculative_algorithm':('none' if args.baseline else 'DSpark compatibility on vLLM0.24 DFlash transport'),
           'runner_pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
           'installed_source_sha256':sources,'responses':[]}
    state['draft_weights_source']={'path':str(draft/'model.safetensors'),
                                    'bytes':(draft/'model.safetensors').stat().st_size}
    server=None

    def save():
        p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')
        print(json.dumps({'phase':state['phase'],'error':state.get('error')}),flush=True)

    def stop(signum,frame):
        if server is not None and server.poll() is None:os.killpg(server.pid,signal.SIGTERM)
        raise RuntimeError(f'Stopped by signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop)
    save()
    try:
        env=dict(os.environ,VLLM_PLUGINS='fl',PYTHONUNBUFFERED='1',
                 TRITON_PRINT_AUTOTUNING='1',
                 PYTHONPATH=str(directory/'hooks'),TIANSHU_DSPARK_COMPAT=('0' if args.baseline else '1'),
                 TIANSHU_DSPARK_RECORD_DIR=str(root/'adapter_records'),
                 TIANSHU_DSPARK_ZERO_VERIFY=('1' if args.zero_verify else '0'),
                 VLLM_FL_USE_FLAGGEMS_ATTN='0',VLLM_FL_ILUVATAR_HYBRID_ATTENTION='0',
                 VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS='0',
                 VLLM_FL_ILUVATAR_PREFILL_BUCKETS='0',VLLM_FL_ILUVATAR_LOGITS_GATHER='0',
                 FLAGGEMS_ILUVATAR_COMPACT_VARLEN='1')
        for name in ['TIANSHU_NATIVE_PREFILL_PROBE','TIANSHU_FA2_TILE_PROBE','TIANSHU_PACK_PROBE_MODE']:
            env.pop(name,None)
        spec={'method':'dflash','model':str(alias),'num_speculative_tokens':7,
              'draft_sample_method':'probabilistic','enforce_eager':True}
        command=['vllm','serve','/workspace/MiniCPM5-2B','--port','9031',
                 '--served-model-name','minicpm','--gpu-memory-utilization','.85',
                 '--max-model-len','131072','--max-num-batched-tokens','2048',
                 '--max-num-seqs',str(args.concurrency),'--enforce-eager','--speculative-config',json.dumps(spec)]
        if args.baseline:
            command=command[:-2]
            spec=None
        if args.target_graphs:
            command.remove('--enforce-eager')
            command+=['--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}']
        state['command']=command;state['speculative_config']=spec
        state['environment_flags']={k:v for k,v in env.items() if k.startswith(
            ('VLLM_FL_','FLAGGEMS_ILUVATAR_','TIANSHU_DSPARK_','TRITON_PRINT_'))}
        with (root/'server.log').open('w') as log:
            server=subprocess.Popen(command,cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='starting',server_pid=server.pid);save()
        limit=time.monotonic()+1200
        while True:
            if server.poll() is not None:raise RuntimeError(f'Server exited during startup: {server.returncode}')
            if time.monotonic()>limit:raise TimeoutError('Startup exceeded20min')
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response:
                    data=json.load(response)
                if any(x['id']=='minicpm' for x in data['data']):break
            except Exception:pass
            time.sleep(3)
        state['phase']='requesting';save()
        bodies=[{'model':'minicpm','prompt':'Compute 17 + 25 and briefly explain the answer.',
                 'max_tokens':96,'temperature':temperature,'top_p':.95,'stream':False}
                for temperature in (0.,1.)]
        if args.diagnostic_prefixes:
            bodies=[{'model':'minicpm','prompt':prompt,'max_tokens':8,
                     'temperature':0.,'top_p':1.,'stream':False,'logprobs':10}
                    for prompt in ('Compute 17 + 25 and briefly explain the answer.',
                                   'Compute 17 + 25 and briefly explain the answer. You are')]
        if args.quick_performance:
            assert args.zero_verify and not args.diagnostic_prefixes
            bodies=bodies[:1]

        def request(body,stream=False):
            request=urllib.request.Request('http://127.0.0.1:9031/v1/completions',data=json.dumps(body).encode(),
                                           headers={'Content-Type':'application/json'})
            start=time.monotonic()
            first=None;text='';usage=None
            with urllib.request.urlopen(request,timeout=900) as response:
                if not stream:data=json.load(response)
                else:
                    for raw in response:
                        line=raw.decode().strip()
                        if not line.startswith('data: ') or line=='data: [DONE]':continue
                        event=json.loads(line[6:])
                        for choice in event.get('choices',[]):
                            chunk=choice.get('text','');text+=chunk
                            if chunk and first is None:first=time.monotonic()-start
                        if event.get('usage'):usage=event['usage']
                    assert usage is not None and first is not None
                    data={'text':text,'usage':usage}
            return {'request':body,'response':data,'seconds':time.monotonic()-start,'ttft_seconds':first}

        for body in bodies:
            state['responses'].append(request(body))
            save()
        if args.quick_performance:
            state['workload']={'input_tokens':4096,'output_tokens':128,'concurrency':args.concurrency,
                               'requests_per_round':args.concurrency*2,'rounds':3,'excluded_rounds':[1],
                               'target_eager':not args.target_graphs,'draft_eager':None if args.baseline else True,
                               'official_benchmark':False}
            state['performance_rounds']=[]
            logpath=root/'server.log'
            for round_index in range(3):
                state['phase']=f'quick_performance_round_{round_index+1}';save()
                before=len(logpath.read_text());started=time.monotonic()
                payloads=[]
                for i in range(args.concurrency*2):
                    rng=random.Random(2026101220+1000*round_index+i)
                    payloads.append({'model':'minicpm','prompt':[rng.randrange(512,120000) for _ in range(4096)],
                                     'max_tokens':128,'ignore_eos':True,'temperature':1.,'top_p':.95,
                                     'stream':True,'stream_options':{'include_usage':True}})
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    responses=list(pool.map(lambda body:request(body,True),payloads))
                elapsed=time.monotonic()-started
                assert all(x['response']['usage']['prompt_tokens']==4096 and
                           x['response']['usage']['completion_tokens']==128 for x in responses)
                state['performance_rounds'].append({'round':round_index+1,'duration_seconds':elapsed,
                    'total_tokens_per_second':len(responses)*4224/elapsed,
                    'mean_ttft_seconds':statistics.mean(x['ttft_seconds'] for x in responses),
                    'successful_requests':len(responses),'responses':responses,
                    'tuning_records':[x for x in logpath.read_text()[before:].splitlines()
                                      if 'Triton autotuning for function' in x]})
                save()
        with urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=3) as response:
            (root/'metrics.txt').write_bytes(response.read())
        records=list((root/'adapter_records').glob('*.json'))
        assert records or args.baseline, 'No actual sequential Markov proposal record'
        state['adapter_records']=[json.loads(p.read_text()) for p in records]
        state['phase']='complete'
    except Exception as error:
        state.update(phase='failed',error=str(error))
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
        state['hardware_after']=subprocess.check_output(['ixsmi'],text=True)
        state['gpu_idle_after']='No running processes found' in state['hardware_after']
        link=alias/'model.safetensors'
        assert link.is_symlink() and link.resolve()==(draft/'model.safetensors').resolve()
        link.unlink()
        state['temporary_weight_symlink_removed_after_cleanup']=True
        state['finished_utc']=datetime.now(timezone.utc).isoformat();save()
        (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' and state['gpu_idle_after'] else 1


if __name__=='__main__':raise SystemExit(main())
