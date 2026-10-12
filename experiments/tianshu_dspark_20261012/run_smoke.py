# SPDX-License-Identifier: Apache-2.0
"""Bring up the DSpark compatibility lane, retain every response and clean up."""
import argparse
import csv
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
import traceback
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
    parser.add_argument('--sample-view',action='store_true')
    parser.add_argument('--verify-tile',action='store_true')
    parser.add_argument('--matched-warmup',action='store_true')
    parser.add_argument('--math-level3',action='store_true')
    parser.add_argument('--prefill-tile',action='store_true')
    parser.add_argument('--output-tokens',type=int,choices=[128,1024],default=128)
    parser.add_argument('--seed-base',type=int,default=2026101220)
    parser.add_argument('--seed-round-stride',type=int,default=1000)
    parser.add_argument('--official-4k',action='store_true')
    parser.add_argument('--quality-results',type=Path)
    parser.add_argument('--standard-results',type=Path)
    parser.add_argument('--fixed-index',action='store_true')
    parser.add_argument('--request-timeout',type=int,default=900)
    parser.add_argument('--fp32-thresholds',action='store_true')
    parser.add_argument('--speculative-tokens',type=int,choices=[1,2,3,7],default=7)
    args=parser.parse_args()
    assert not args.verify_tile or args.zero_verify
    assert not args.prefill_tile or args.zero_verify
    assert not args.math_level3 or (not args.baseline and args.zero_verify and
                                   not args.quick_performance and not args.diagnostic_prefixes)
    assert not args.official_4k or (not args.baseline and args.zero_verify and args.target_graphs and
        not args.quick_performance and not args.math_level3 and args.quality_results and args.standard_results)
    assert not (args.math_level3 or args.official_4k) or args.fp32_thresholds, \
        'CoreX FP64 rejection threshold path fails CPU oracle; validated FP32 compatibility is required'
    assert args.baseline or not args.quick_performance or args.fp32_thresholds, \
        'Stochastic speculative performance requires validated FP32 rejection thresholds'
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    assert not (root/'state.json').exists()
    lease=Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    deadline=datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
    remaining=int((deadline-datetime.now(timezone.utc)).total_seconds())
    assert remaining>0,'Reservation has ended'
    signal.alarm(min(10800 if args.math_level3 else 7200 if args.official_4k else 3600,remaining))
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
    frozen_directory=snapshot/'experiments'/directory.name
    for p in directory.rglob('*.py'):
        if 'results' in p.parts:continue
        target=frozen_directory/p.relative_to(directory);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    generator=directory.parent/'tianshu_native_prefill_20261012/zero_initialization.py'
    frozen_generator=snapshot/'experiments'/'tianshu_native_prefill_20261012'/generator.name
    frozen_generator.parent.mkdir(parents=True,exist_ok=True)
    frozen_generator.write_bytes(generator.read_bytes())
    installed=Path('/usr/local/lib/python3.12/site-packages')
    sources={}
    for rel in ['vllm_fl/worker/worker.py','vllm_fl/worker/model_runner.py',
                'vllm/config/speculative.py','vllm/v1/spec_decode/dflash.py',
                'vllm/v1/spec_decode/llm_base_proposer.py','vllm/model_executor/models/qwen3_dflash.py']:
        p=installed/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
        target=snapshot/'installed'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    state={'phase':'preparing','official_benchmark':args.official_4k,'accuracy_test':args.math_level3,
           'speculative_algorithm':('none' if args.baseline else 'DSpark compatibility on vLLM0.24 DFlash transport'),
           'runner_pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
           'installed_source_sha256':sources,'responses':[]}
    state['draft_weights_source']={'path':str(draft/'model.safetensors'),
                                    'bytes':(draft/'model.safetensors').stat().st_size}
    server=evaluator=None

    def save():
        p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')
        print(json.dumps({'phase':state['phase'],'error':state.get('error')}),flush=True)

    def stop(signum,frame):
        if server is not None and server.poll() is None:os.killpg(server.pid,signal.SIGTERM)
        if evaluator is not None and evaluator.poll() is None:os.killpg(evaluator.pid,signal.SIGTERM)
        raise RuntimeError(f'Stopped by signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop)
    save()
    try:
        env=dict(os.environ,VLLM_PLUGINS='fl',PYTHONUNBUFFERED='1',
                 TRITON_PRINT_AUTOTUNING='1',
                 PYTHONPATH=str(frozen_directory/'hooks'),TIANSHU_DSPARK_COMPAT=('0' if args.baseline else '1'),
                 TIANSHU_DSPARK_RECORD_DIR=str(root/'adapter_records'),
                 TIANSHU_DSPARK_ZERO_VERIFY=('1' if args.zero_verify else '0'),
                 TIANSHU_DSPARK_SAMPLE_VIEW=('1' if args.sample_view else '0'),
                 TIANSHU_DSPARK_VERIFY_TILE=('1' if args.verify_tile else '0'),
                 TIANSHU_DSPARK_MATCHED_WARMUP=('1' if args.matched_warmup else '0'),
                 TIANSHU_DSPARK_PREFILL_TILE=('1' if args.prefill_tile else '0'),
                 TIANSHU_DSPARK_FIXED_INDEX=('1' if args.fixed_index else '0'),
                 TIANSHU_DSPARK_FP32_THRESHOLDS=('1' if args.fp32_thresholds else '0'),
                 VLLM_FL_USE_FLAGGEMS_ATTN='0',VLLM_FL_ILUVATAR_HYBRID_ATTENTION='0',
                 VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS='0',
                 VLLM_FL_ILUVATAR_PREFILL_BUCKETS='0',VLLM_FL_ILUVATAR_LOGITS_GATHER='0',
                 FLAGGEMS_ILUVATAR_COMPACT_VARLEN='1')
        for name in ['TIANSHU_NATIVE_PREFILL_PROBE','TIANSHU_FA2_TILE_PROBE','TIANSHU_PACK_PROBE_MODE']:
            env.pop(name,None)
        spec={'method':'dflash','model':str(alias),'num_speculative_tokens':args.speculative_tokens,
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
        state['frozen_hook_directory']=str(frozen_directory/'hooks')
        state['environment_flags']={k:v for k,v in env.items() if k.startswith(
            ('VLLM_FL_','FLAGGEMS_ILUVATAR_','TIANSHU_DSPARK_','TRITON_PRINT_'))}
        if args.official_4k:
            quality=json.loads((args.quality_results/'accuracy_audit.json').read_text())
            quality_state=json.loads((args.quality_results/'result.json').read_text())
            standard=json.loads((args.standard_results/'comparison.json').read_text())
            assert quality['complete'] and quality['total']==105 and quality['accuracy']>.95
            assert quality['official_requirement_passed'] and not quality['reused_accuracy']
            assert standard['promising'] and standard['measured_tuning_clean']
            assert quality['installed_source_sha256']==sources
            def normalized_server_command(values):
                values=list(values);index=values.index('--speculative-config')+1
                specification=json.loads(values[index]);specification['model']='same_readonly_draft'
                values[index]=json.dumps(specification,sort_keys=True)
                return values
            assert normalized_server_command(quality['server_command'])==normalized_server_command(command)
            relevant=('TIANSHU_DSPARK_COMPAT','TIANSHU_DSPARK_ZERO_VERIFY','TIANSHU_DSPARK_SAMPLE_VIEW',
                      'TIANSHU_DSPARK_VERIFY_TILE','TIANSHU_DSPARK_PREFILL_TILE','TIANSHU_DSPARK_MATCHED_WARMUP',
                      'TIANSHU_DSPARK_FIXED_INDEX','TIANSHU_DSPARK_FP32_THRESHOLDS')
            assert all(quality['environment_flags'].get(k,'0')==env.get(k,'0') for k in relevant)
            quality_directory=Path(quality_state['frozen_hook_directory']).parent
            for rel in ('hooks/sitecustomize.py','tianshu_dspark_adapter/model.py','tianshu_dspark_adapter/proposer.py',
                        'fixed_index.py','rejection_compat.py'):
                assert (frozen_directory/rel).read_bytes()==(quality_directory/rel).read_bytes()
            quality_generator=quality_directory.parent/'tianshu_native_prefill_20261012/zero_initialization.py'
            assert frozen_generator.read_bytes()==quality_generator.read_bytes()
            bench=directory.parents[1]/'benchmarks/benchmark_throughput_serve.py'
            benchmark_sha='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
            assert hashlib.sha256(bench.read_bytes()).hexdigest()==benchmark_sha
            state.update(official_script_sha256=benchmark_sha,runs=4,skip_first=1,
                         test_cases=[[4096,1024,64,256]],independent_math={
                             'correct':quality['correct'],'total':105,'accuracy':quality['accuracy']},
                         standard_comparison=standard)
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
        if args.quick_performance or args.math_level3 or args.official_4k:
            assert args.zero_verify and not args.diagnostic_prefixes
            bodies=bodies[:1]

        def request(body,stream=False):
            request=urllib.request.Request('http://127.0.0.1:9031/v1/completions',data=json.dumps(body).encode(),
                                           headers={'Content-Type':'application/json'})
            start=time.monotonic()
            first=None;text='';usage=None
            with urllib.request.urlopen(request,timeout=args.request_timeout) as response:
                if not stream:data=json.load(response)
                else:
                    for raw in response:
                        line=raw.decode().strip()
                        if not line.startswith('data: ') or line=='data: [DONE]':continue
                        event=json.loads(line[6:])
                        if 'error' in event:
                            raise RuntimeError(f'Server streaming error: {event["error"]}')
                        for choice in event.get('choices',[]):
                            chunk=choice.get('text','');text+=chunk
                            if chunk and first is None:first=time.monotonic()-start
                        if event.get('usage'):usage=event['usage']
                    assert usage is not None and first is not None,'Incomplete completion stream: missing usage or first token'
                    data={'text':text,'usage':usage}
            return {'request':body,'response':data,'seconds':time.monotonic()-start,'ttft_seconds':first}

        for body in bodies:
            state['responses'].append(request(body))
            save()
        if args.quick_performance:
            state['workload']={'input_tokens':4096,'output_tokens':args.output_tokens,'concurrency':args.concurrency,
                               'requests_per_round':args.concurrency*2,'rounds':3,'excluded_rounds':[1],
                               'target_eager':not args.target_graphs,'draft_eager':None if args.baseline else True,
                               'seed_base':args.seed_base,'seed_round_stride':args.seed_round_stride,
                               'official_benchmark':False}
            state['performance_rounds']=[]
            logpath=root/'server.log'
            for round_index in range(3):
                state['phase']=f'quick_performance_round_{round_index+1}';save()
                with urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=3) as response:
                    (root/f'metrics_before_round_{round_index+1}.txt').write_bytes(response.read())
                before=len(logpath.read_text());started=time.monotonic()
                payloads=[]
                for i in range(args.concurrency*2):
                    rng=random.Random(args.seed_base+args.seed_round_stride*round_index+i)
                    payloads.append({'model':'minicpm','prompt':[rng.randrange(512,120000) for _ in range(4096)],
                                     'max_tokens':args.output_tokens,'ignore_eos':True,'temperature':1.,'top_p':.95,
                                     'stream':True,'stream_options':{'include_usage':True}})
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    responses=list(pool.map(lambda body:request(body,True),payloads))
                elapsed=time.monotonic()-started
                with urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=3) as response:
                    (root/f'metrics_after_round_{round_index+1}.txt').write_bytes(response.read())
                assert all(x['response']['usage']['prompt_tokens']==4096 and
                           x['response']['usage']['completion_tokens']==args.output_tokens for x in responses)
                state['performance_rounds'].append({'round':round_index+1,'duration_seconds':elapsed,
                    'total_tokens_per_second':len(responses)*(4096+args.output_tokens)/elapsed,
                    'mean_ttft_seconds':statistics.mean(x['ttft_seconds'] for x in responses),
                    'successful_requests':len(responses),'responses':responses,
                    'tuning_records':[x for x in logpath.read_text()[before:].splitlines()
                                      if 'Triton autotuning for function' in x]})
                save()
        if args.math_level3:
            evaluation=['evalscope','eval','--model','minicpm',
                '--api-url','http://127.0.0.1:9031/v1/chat/completions','--api-key','EMPTY',
                '--eval-type','openai_api','--datasets','math_500',
                '--dataset-args','{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}',
                '--eval-batch-size','8','--timeout','3600',
                '--generation-config','{"temperature":1.0,"top_p":0.95,"max_tokens":32768}',
                '--work-dir',str(root/'evalscope'),'--ignore-errors']
            state['evaluation_command']=evaluation
            with (root/'evalscope.log').open('w') as log:
                evaluator=subprocess.Popen(evaluation,cwd='/workspace',env=env,stdout=log,
                                           stderr=subprocess.STDOUT,start_new_session=True)
            state.update(phase='evaluating',evaluator_pid=evaluator.pid);save()
            state['evalscope_exit_code']=evaluator.wait()
            assert state['evalscope_exit_code']==0,'EvalScope failed'
        if args.official_4k:
            command=['python3','-u',str(bench),'--model','/workspace/MiniCPM5-2B',
                     '--served-model-name','minicpm','--port','9031',
                     '--test-cases','[[4096,1024,64,256]]']
            state['benchmark_command']=command
            boundary=len((root/'server.log').read_text())
            with (root/'benchmark.log').open('w') as log:
                evaluator=subprocess.Popen(command,cwd=root,env=env,stdout=log,
                                           stderr=subprocess.STDOUT,start_new_session=True)
            state.update(phase='benchmarking',benchmark_pid=evaluator.pid);save()
            while evaluator.poll() is None:
                files=list((root/'benchmark_results').glob('raw_runs_*.csv'))
                if files:
                    assert len(files)==1
                    with files[0].open(newline='') as file:rows=list(csv.DictReader(file))
                    if state.get('completed_runs')!=len(rows):
                        state.update(completed_runs=len(rows),raw_runs=rows);save()
                time.sleep(5)
            state['benchmark_exit_code']=evaluator.returncode;assert evaluator.returncode==0
            raw=list((root/'benchmark_results').glob('raw_runs_*.csv'))
            summaries=list((root/'benchmark_results').glob('summary_*.csv'))
            assert len(raw)==len(summaries)==1
            with raw[0].open(newline='') as file:rows=list(csv.DictReader(file))
            with summaries[0].open(newline='') as file:summary=list(csv.DictReader(file))
            assert len(rows)==4 and len(summary)==1
            for row in rows:
                assert row['Run Status']=='SUCCESS' and int(float(row['Successful Requests']))==256
                assert [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
                assert int(float(row['Total Input Tokens']))==1048576 and int(float(row['Total Output Tokens']))==262144
            for key in ('Total tok/s','Output tok/s','Mean TTFT (ms)'):
                assert abs(float(summary[0][key])-statistics.mean(float(x[key]) for x in rows[1:]))<.011
            assert hashlib.sha256(bench.read_bytes()).hexdigest()==benchmark_sha
            state.update(completed_runs=4,raw_runs=rows,official_summary=summary[0],
                benchmark_runtime_tuning_records=[x for x in (root/'server.log').read_text()[boundary:].splitlines()
                                                 if 'Triton autotuning for function' in x])
        with urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=3) as response:
            (root/'metrics.txt').write_bytes(response.read())
        records=list((root/'adapter_records').glob('*.json'))
        assert records or args.baseline, 'No actual sequential Markov proposal record'
        state['adapter_records']=[json.loads(p.read_text()) for p in records]
        state['phase']='evaluation_finished_pending_report_audit' if args.math_level3 else 'complete'
    except Exception as error:
        state.update(phase='failed',error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
    finally:
        if evaluator is not None and evaluator.poll() is None:
            os.killpg(evaluator.pid,signal.SIGTERM)
            try:evaluator.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(evaluator.pid,signal.SIGKILL);evaluator.wait(timeout=5)
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
        state['hardware_after']=subprocess.check_output(['ixsmi'],text=True)
        (root/'hardware_after.txt').write_text(state['hardware_after'])
        state['gpu_idle_after']='No running processes found' in state['hardware_after']
        link=alias/'model.safetensors'
        assert link.is_symlink() and link.resolve()==(draft/'model.safetensors').resolve()
        link.unlink()
        state['temporary_weight_symlink_removed_after_cleanup']=True
        state['finished_utc']=datetime.now(timezone.utc).isoformat();save()
        (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase'] in ('complete','evaluation_finished_pending_report_audit') and state['gpu_idle_after'] else 1


if __name__=='__main__':raise SystemExit(main())
