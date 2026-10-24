# SPDX-License-Identifier: Apache-2.0
"""One fresh API process, exact original serve argv, bounded evaluation, cleanup."""
import argparse,json,os,signal,socket,subprocess,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from candidate_profile import prepare,digest,read,frozen_check
from audit import audit_model,audit_math
BASE=Path(__file__).resolve().parent
DEADLINE=datetime.fromisoformat('2026-10-24T15:58:00+00:00')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('candidate','control'),default='candidate')
    parser.add_argument('--stage',choices=('math',),default='math')
    parser.add_argument('--batch',type=int,choices=(8,64),default=64)
    args=parser.parse_args()
    assert args.stage=='math' or args.batch==64
    assert args.mode in ('candidate','control')
    root=args.output.resolve()
    assert root.is_relative_to('/dev/shm/tianshu_r5_BM8_runs_20261024') and not root.exists()
    root.mkdir(parents=True)
    state=dict(phase='preparing',mode='candidate',bm8_mode=args.mode,stage=args.stage,batch=args.batch,
        runner_pid=os.getpid(),adopted=False,official=args.stage=='official',started_at=datetime.now(timezone.utc).isoformat())
    server=evaluator=None
    def save():
        tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json')
        print(json.dumps({k:state.get(k) for k in ('phase','server_pid','evaluator_pid','error','correct')}),flush=True)
    def terminate(process):
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
    def stop(signum,frame): raise TimeoutError('signal/deadline '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
    save()
    try:
        remaining=int((DEADLINE-datetime.now(timezone.utc)).total_seconds())
        assert remaining>3600
        signal.alarm(remaining)
        command,env,profile,sources=prepare(root,args.mode)
        if args.stage=='math' and args.batch==64:
            profile['TIANSHU_FUTURE_KV_MODE']='control';env.update(profile)
        full={**profile,'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']}
        assert len(full)==67 and len(command)==13
        controllers={str(p):digest(p) for p in (BASE/'run_model.py',BASE/'audit.py')}
        state.update(server_command=command,profile_environment=profile,frozen_sources=sources,
            controller_sources=controllers,full_environment=full,
            deadline=DEADLINE.isoformat(),fresh_API_per_batch=True,only_execution_delta='M64 3D quantized KV reader BM16 Q2 grid96 to BM8 Q1 grid64; full r5 retained')
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0
        hardware=subprocess.check_output(['ixsmi'],text=True)
        (root/'hardware_before.txt').write_text(hardware)
        assert 'No running processes found' in hardware
        with (root/'server.log').open('w') as log:
            server=subprocess.Popen(command,cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='starting_server',server_pid=server.pid);save()
        limit=time.monotonic()+2400
        while True:
            assert server.poll() is None,'Server exited during startup'
            assert time.monotonic()<limit,'startup timeout'
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response: models=json.load(response)
                if any(m['id']=='minicpm' for m in models['data']):break
            except Exception:pass
            time.sleep(3)
        observed=dict(v.decode().split('=',1) for v in Path(f'/proc/{server.pid}/environ').read_bytes().split(b'\0') if b'=' in v)
        assert {k:observed.get(k) for k in full}==full
        observed_command=Path(f'/proc/{server.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        state['observed_commandline']=observed_command
        (root/'observed_environment.json').write_text(json.dumps(full,indent=2)+'\n')
        audit_model(root,state)
        def metrics(name):
            with urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=10) as response:(root/name).write_bytes(response.read())
        metrics('before_metrics.txt')
        if args.stage=='math':
            evaluation=['evalscope','eval','--model','minicpm','--api-url','http://127.0.0.1:9031/v1/chat/completions',
                '--api-key','EMPTY','--eval-type','openai_api','--datasets','math_500','--dataset-args',
                '{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}',
                '--eval-batch-size',str(args.batch),'--timeout','3600','--generation-config',
                '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}','--work-dir',str(root/'evalscope'),'--ignore-errors']
        else:
            benchmark=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
            assert digest(benchmark)=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
            count=256 if args.stage=='official' else 64
            evaluation=['python3','-u',str(benchmark),'--model','/workspace/MiniCPM5-2B','--served-model-name','minicpm',
                '--port','9031','--test-cases',json.dumps([[4096,1024,64,count]],separators=(',',':'))]
        state.update(evaluation_command=evaluation,performance_preexcluded_rounds=[1] if args.stage!='math' else [])
        evalenv=dict(env);evalenv.pop('PYTHONPATH',None)
        with (root/'evaluation.log').open('w') as log:
            evaluator=subprocess.Popen(evaluation,cwd=str(root),env=evalenv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='evaluating',evaluator_pid=evaluator.pid);save()
        while evaluator.poll() is None:
            assert server.poll() is None,'API server exited'
            time.sleep(5)
        state['evaluation_exit_code']=evaluator.returncode
        assert evaluator.returncode==0
        metrics('after_metrics.txt')
        frozen_check(sources);frozen_check(controllers)
        state['phase']='evaluation_finished_pending_audit'
    except BaseException as error:
        state.update(phase='failed',error=repr(error))
    finally:
        terminate(evaluator);terminate(server);signal.alarm(0)
        for _ in range(60):
            hardware=subprocess.check_output(['ixsmi'],text=True)
            (root/'hardware_after.txt').write_text(hardware)
            if 'No running processes found' in hardware:break
            time.sleep(1)
        else:state.update(phase='cleanup_failed',error='GPU did not become idle')
        save()
    if state['phase']=='evaluation_finished_pending_audit':
        try:
            profile=audit_model(root,state,runtime=True)
            (root/'profile_audit.json').write_text(json.dumps(profile,indent=2)+'\n')
            if args.stage=='math':
                quality=audit_math(root,state)
                (root/'accuracy_audit.json').write_text(json.dumps(quality,indent=2)+'\n')
                state.update(correct=quality['correct'],total=105,accuracy=quality['accuracy'],accuracy_passed=quality['above95'])
            state['phase']='complete'
        except BaseException as error:state.update(phase='audit_failed',error=repr(error))
    save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
