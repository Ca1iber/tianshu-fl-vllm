# SPDX-License-Identifier: Apache-2.0
"""Fresh official Level3 evaluation of budget4096/83 buckets with B64S8."""
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
    parser.add_argument('--service-root',type=Path,required=True)
    parser.add_argument('--adaptive',choices=['1'],default='1')
    parser.add_argument('--candidate',choices=['decode-coherent'],required=True)
    parser.add_argument('--max-batched-tokens',type=int,choices=[4096],default=4096)
    parser.add_argument('--buckets',choices=['0','1'],default='0')
    parser.add_argument('--gather',choices=['0','1'],default='0')
    parser.add_argument('--bucket-trace',choices=['0','2'],default='0')
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    if args.buckets=='1' or args.gather=='1' or args.bucket_trace!='0':
        assert args.candidate=='decode-coherent' and args.max_batched_tokens==4096
        assert args.gather!='1' or args.buckets=='1'
        assert args.bucket_trace=='0' or args.buckets==args.gather=='1'
    if args.candidate not in ('native-prefill','decode-coherent') and args.max_batched_tokens!=2048:
        parser.error('GEMM/sampler candidates require their independently controlled budget2048 profile')
    if (root/'result.json').exists():raise RuntimeError('Refusing to overwrite a completed evaluation')
    state={'buckets':args.buckets,'gather':args.gather,'max_batched_tokens':args.max_batched_tokens,'phase':'preparing','candidate':f'hybrid_native_prefill_L0_budget{args.max_batched_tokens}_KV16_w1_guarded_B64S8','reused_original_accuracy':False,
           'started_utc':datetime.now(timezone.utc).isoformat(),'runner_pid':os.getpid()}
    comparison=json.loads((args.service_root/'comparison.json').read_text())
    service=json.loads((args.service_root/'candidate/result.json').read_text())
    assert comparison['complete'] and comparison['gain_percent']>=2 and service['phase']=='complete'
    own_hook=Path(__file__).resolve().parent/'hooks/sitecustomize.py'
    assert hashlib.sha256(own_hook.read_bytes()).hexdigest()==service['active_hook']['sha256']
    assert service["lowwarps_mode"] == "candidate" and service["actual_budget"] == 4096
    assert comparison['control_budget']==2048 and comparison['candidate_budget']==4096
    assert comparison['candidate_decode_warps']==1 and comparison['bucket_points']==[75,83]
    assert service['actual_bucket_points']==83 and service['actual_extended_policy_verified']
    assert service['actual_sql_mode_verified'] and service['sql_reflection_enabled']
    assert comparison['segment_policy']=={'B64':8,'others':'unchanged'}
    assert args.buckets == args.gather == "1" and args.bucket_trace == "2"
    assert len(service["short_4k_rounds"]) == 5 and comparison["excluded_rounds"] == [1, 2]
    assert min(comparison["candidate_rounds"][2:]) > max(comparison["control_rounds"][2:])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in comparison["frozen_sources"].items())
    state['matched_service_root']=str(args.service_root)
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
            VLLM_FL_ILUVATAR_PREFILL_BUCKETS=args.buckets,VLLM_FL_ILUVATAR_LOGITS_GATHER=args.gather,TRITON_PRINT_AUTOTUNING='1',
            VLLM_FL_ILUVATAR_HYBRID_ATTENTION='1',VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS=args.adaptive,VLLM_FL_ILUVATAR_HYBRID_STATS_PATH=str(root/'routing_stats.json'))
        env.pop('PYTHONPATH',None);env.pop('TIANSHU_PACK_PROBE_MODE',None)
        base=Path(__file__).resolve().parents[1]
        candidates={'small-gemm':('tianshu_small_gemm_20261012','TIANSHU_SMALL_GEMM_PROBE','TIANSHU_SMALL_GEMM_RECORD_DIR','small_mm_records','mm_*.json'),
                    'prefill-gemm':('tianshu_prefill_gemm_20261012','TIANSHU_PREFILL_GEMM_PROBE','TIANSHU_PREFILL_GEMM_RECORD_DIR','prefill_mm_records','mm_*.json'),
                    'sampler':('tianshu_sampler_fusion_20261012','TIANSHU_SAMPLER_FUSION_PROBE','TIANSHU_SAMPLER_RECORD_DIR','sampler_records','sampler_*.json'),
                    'native-prefill':('tianshu_native_prefill_20261012','TIANSHU_NATIVE_PREFILL_PROBE','TIANSHU_NATIVE_PREFILL_RECORD_DIR','native_prefill_records','native_prefill_*.json'),
                    'decode-coherent':('tianshu_budget4096_buckets_20261014','TIANSHU_DECODE_COHERENT_PROBE','TIANSHU_DECODE_COHERENT_RECORD_DIR','decode_coherent_records','producer_*.json')}
        folder,flag,record_flag,record_dir,pattern=candidates[args.candidate]
        for _, candidate_flag, candidate_record_flag, _, _ in candidates.values():
            env.pop(candidate_flag, None)
            env.pop(candidate_record_flag, None)
        hook=base/folder/'hooks/sitecustomize.py'
        state['experimental_candidate']=args.candidate
        env[flag]='1';env[record_flag]=str(root/record_dir)
        if args.candidate=='decode-coherent':
            env['TIANSHU_DECODE_LAUNCH_PROBE']='1'
            env['TIANSHU_DECODE_LAUNCH_RECORD_DIR']=str(root/'legacy_decode_fallback_records')
            env['TIANSHU_NATIVE_PREFILL_PROBE']='1'
            env['TIANSHU_NATIVE_PREFILL_RECORD_DIR']=str(root/'native_prefill_records')
        env['PYTHONPATH']=str(hook.parent)
        env['TIANSHU_FA2_TILE_PROBE']='1'
        env['TIANSHU_FA2_TILE_RECORD_DIR']=str(root/'tile_records')
        state['tile_hook']={'path':str(hook),'sha256':hashlib.sha256(hook.read_bytes()).hexdigest()}
        state['tile_configuration']={'BLOCK_M':128,'BLOCK_N':128,'num_warps':8,'num_stages':2}
        state['native_prefill_configuration']={'BLOCK_M':256,'TILE_SIZE':128,'num_warps':16,'num_stages':1,'initialization':'L0'}
        if args.bucket_trace=='2':
            trace=base/'tianshu_buckets75_recheck_20261012/joint_hooks/sitecustomize.py'
            env['PYTHONPATH']=str(trace.parent)
            env['TIANSHU_BUCKET_RECHECK_TRACE']='1'
            env['TIANSHU_BUCKET_TRACE_DIR']=str(root/'bucket_trace_records')
            state['bucket_trace_hook']={'path':str(trace),'sha256':hashlib.sha256(trace.read_bytes()).hexdigest()}
        command=['vllm','serve','/workspace/MiniCPM5-2B','--port','9031',
            '--compilation-config','{"cudagraph_mode": "FULL_DECODE_ONLY"}',
            '--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
        command += ['--max-num-batched-tokens',str(args.max_batched_tokens)]
        command += ['--long-prefill-token-threshold', '0']
        assert command == service['command']
        state['server_command']=command
        source=Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker/iluvatar_hybrid_attention.py')
        state['hybrid_source']=str(source);state['hybrid_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        adaptive=source.with_name('iluvatar_adaptive_decode.py');state['adaptive_flag']=args.adaptive;state['adaptive_source']=str(adaptive);state['adaptive_sha256']=hashlib.sha256(adaptive.read_bytes()).hexdigest()
        snapshots=root/'source_snapshot';snapshots.mkdir(exist_ok=True)
        (snapshots/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        (snapshots/'candidate_sitecustomize.py').write_bytes(hook.read_bytes())
        tile_hook=base/'tianshu_fa2_tiles_20261012/hooks/sitecustomize.py'
        (snapshots/'tile_sitecustomize.py').write_bytes(tile_hook.read_bytes())
        state['candidate_hook']=dict(state['tile_hook'])
        state['tile_hook']={'path':str(tile_hook),'sha256':hashlib.sha256(tile_hook.read_bytes()).hexdigest()}
        kernel=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py') if args.candidate=='decode-coherent' else base/('tianshu_sampler_fusion_20261012/experimental_sampler.py' if args.candidate=='sampler' else
                    'tianshu_native_prefill_20261012/zero_initialization.py' if args.candidate=='native-prefill' else
                    'tianshu_splitk_20261012/experimental_mm.py')
        (snapshots/kernel.name).write_bytes(kernel.read_bytes())
        state['experimental_kernel']={'path':str(kernel),'sha256':hashlib.sha256(kernel.read_bytes()).hexdigest()}
        if args.candidate=='decode-coherent':
            native_hook=base/'tianshu_native_prefill_20261012/hooks/sitecustomize.py'
            generator=base/'tianshu_native_prefill_20261012/zero_initialization.py'
            (snapshots/'native_prefill_sitecustomize.py').write_bytes(native_hook.read_bytes())
            (snapshots/'zero_initialization.py').write_bytes(generator.read_bytes())
            state['native_prefill_hook']={'path':str(native_hook),'sha256':hashlib.sha256(native_hook.read_bytes()).hexdigest()}
            state['zero_generator']={'path':str(generator),'sha256':hashlib.sha256(generator.read_bytes()).hexdigest()}
        for name in ('worker.py','iluvatar_hybrid_attention.py','iluvatar_adaptive_decode.py'):
            current=source.with_name(name);contents=current.read_bytes();(snapshots/name).write_bytes(contents)
            if name=='worker.py':state['worker_sha256']=hashlib.sha256(contents).hexdigest()
        if args.buckets=='1':
            state['bucket_sources']={}
            for name in ('model_runner.py','iluvatar_prefill_buckets.py','iluvatar_logits_gather.py'):
                current=source.with_name(name);contents=current.read_bytes();(snapshots/name).write_bytes(contents)
                state['bucket_sources'][name]={'path':str(current),'sha256':hashlib.sha256(contents).hexdigest()}
        if args.bucket_trace=='2':
            (snapshots/'bucket_trace_sitecustomize.py').write_bytes(trace.read_bytes())
            (snapshots/'bucket_component_sitecustomize.py').write_bytes((base/'tianshu_buckets75_recheck_20261012/hooks/sitecustomize.py').read_bytes())
        env['PYTHONPATH']=str(hook.parent)
        env['TIANSHU_KV16_SERVICE_MODE']='candidate'
        env['TIANSHU_SEGMENT_POLICY_PROBE']='1'
        env['TIANSHU_SEGMENT_POLICY_RECORD_DIR']=str(root/'segment_policy_records')
        env['TIANSHU_SQL_REFLECTION_PROBE']='1'
        env['TIANSHU_SQL_REFLECTION_RECORD_DIR']=str(root/'sql_reflection_records')
        env['TIANSHU_EXTENDED_BUCKET_PROBE']='1'
        env['TIANSHU_EXTENDED_BUCKET_RECORD_DIR']=str(root/'extended_bucket_records')
        env["TIANSHU_TOPP_LAUNCH_PROBE"] = "1"
        env["TIANSHU_TOPP_LAUNCH_RECORD_DIR"] = str(root / "topp_launch_records")
        kv16_hook = base / "tianshu_decode_kv16_w2_20261014/hooks/sitecustomize.py"
        decode_hook = kv16_hook.with_name("decode_sitecustomize.py")
        own_sources = [Path(__file__), Path(__file__).with_name("audit_math_eval.py"), hook, kv16_hook, decode_hook,
                       Path(__file__).with_name('extend_buckets.py'), Path(__file__).with_name('audit_profile.py'),
                       base/'tianshu_sql_workflow_20261014/hooks/sitecustomize.py',
                       base/'tianshu_sql_workflow_20261014/reflect_one.py',
                       base/'tianshu_kv16_segments_20261014/hooks/sitecustomize.py']
        state["segment_sources"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in own_sources}
        assert len(state["segment_sources"]) == 10
        for p in own_sources:
            dest = snapshots / "segment_sources" / p.resolve().relative_to(base)
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(p.read_bytes())
        for name,meta in service['installed_sources'].items():
            assert hashlib.sha256(source.with_name(name).read_bytes()).hexdigest()==meta['sha256']
        state['installed_sources']=service['installed_sources']
        state['environment_flags']={k:v for k,v in env.items() if k.startswith(('VLLM_FL_','TIANSHU_','TRITON_PRINT_'))}
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
        if not records and args.candidate not in ('native-prefill','decode-coherent'):raise RuntimeError('Actual model did not use the experimental tile')
        state['tile_routing_records']=[json.loads(p.read_text()) for p in records]
        routes=[json.loads(p.read_text()) for p in (root/record_dir).glob(pattern)]
        if not routes:raise RuntimeError('Experimental candidate did not route in the evaluated model')
        if args.candidate in ('small-gemm','prefill-gemm') and len({tuple(x['shape']) for x in routes})!=3:
            raise RuntimeError('Evaluated GEMM candidate did not route all three checked shapes')
        state['candidate_routing_records']=routes
        if args.candidate=='decode-coherent':
            native_routes=[json.loads(p.read_text()) for p in (root/'native_prefill_records').glob('native_prefill_*.json')]
            assert native_routes and all(x['kernel_sha256']==state['zero_generator']['sha256'] for x in native_routes)
            for x in routes:
                tile = 16 if x["batch"] in (8,16,32,40,48,56,64) else 32
                warps = 1 if tile == 16 else 2 if 8 <= x["batch"] <= 64 else 4
                assert x["configuration"] == dict(TILE_SIZE=tile, num_warps=warps, num_stages=1)
                assert x["compiled_metadata"]["num_warps"] == warps
            assert any(x["batch"] == 8 and x["configuration"] == dict(TILE_SIZE=16,num_warps=1,num_stages=1) for x in routes)
            topp = [json.loads(p.read_text()) for p in (root / "topp_launch_records").glob("route_*.json")]
            assert topp and all(x["configuration"] == dict(num_warps=16, num_stages=1) for x in topp)
            state["topp_launch_records"] = topp
            policy=[json.loads(p.read_text()) for p in (root/'segment_policy_records').glob('choose_*.json')]
            assert policy and any(x['batch']==64 and x['selected_segments']==8 for x in policy)
            assert all(x['selected_segments']==(8 if x['batch']==64 else x['baseline_segments']) for x in policy)
            assert any(x['batch']==64 and x['segments']==8 for x in routes)
            state['segment_policy_records']=policy
            state['B64_binding_scope']='Startup/capture plus prior real4Kservice; official Math concurrency8 does not prove B64 question coverage'
            state['fresh_math_for_B64S8_segments']=True
            state["fresh_math_for_decode_kv16_w1"] = True
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in state["segment_sources"].items())
            assert all(x['generated_source_sha256']=='4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e' for x in routes)
            reducers=[json.loads(p.read_text()) for p in (root/'decode_coherent_records').glob('reducer_*.json')]
            assert reducers and all(x['actual_tile']==(16 if x['batch'] in (8,16,32,40,48,56,64) else 32) for x in reducers)
            assert {(x['batch'],x['segments']) for x in routes} == {(x['batch'],x['segments']) for x in reducers}
            state['decode_coherent_reducer_records']=reducers
            state['fresh_math_for_coherent_decode']=True
            state['native_prefill_routing_records']=native_routes
        state['phase']='evaluation_finished_pending_report_audit'
    except StopRequested as error:state.update(phase='stopped',error=str(error))
    except BaseException as error:state.update(phase='failed',error=str(error),traceback=traceback.format_exc())
    finally:
        terminate(evaluator);terminate(server)
        state['finished_utc']=datetime.now(timezone.utc).isoformat();save()
        (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
    if state['phase'] == 'evaluation_finished_pending_report_audit':
        subprocess.run(["python3", str(Path(__file__).with_name("audit_math_eval.py")), "--output", str(root)], check=True, capture_output=True)
        subprocess.run(["python3", str(Path(__file__).resolve().parents[1] / "archive_completed_pair.py"),
                        "--single-probe", "--source", str(root), "--completion-file", "accuracy_audit.json",
                        "--label", "budget4096_buckets83_S8_math_20261014"], check=True)
        return 0
    return 1


if __name__=='__main__':raise SystemExit(main())
