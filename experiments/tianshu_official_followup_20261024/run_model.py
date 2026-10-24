# SPDX-License-Identifier: Apache-2.0
"""Original official 4K/16K script, final selected profile, bounded fresh API."""
import argparse,importlib.util,json,os,signal,socket,subprocess,sys,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent
EXP=BASE.parent
RUNS=Path('/dev/shm/tianshu_official_followup_20261024')
DEADLINE=datetime.fromisoformat('2026-10-24T15:58:00+00:00')
BENCH=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=('r5','bm8'),required=True);p.add_argument('--case',choices=('4k','16k'),required=True)
    args=p.parse_args();root=args.output.resolve()
    assert root.is_relative_to(RUNS) and not root.exists();root.mkdir(parents=True)
    model=EXP/('tianshu_r5_BM8_KV_20261024' if args.profile=='bm8' else 'tianshu_W4_QKVO64_fitted_service_20261024')
    sys.path.insert(0,str(model))
    from candidate_profile import prepare,digest,read,frozen_check
    audit=load('_official_selected_'+args.profile+'_audit',model/'audit.py')
    r5=audit.parent if args.profile=='bm8' else audit
    audit_long=load('_official_real16K_audit',BASE/'audit16k.py')
    case=[4096,1024,64,256] if args.case=='4k' else [16384,1024,64,128]
    state=dict(phase='preparing',profile_name=args.profile,mode='candidate',batch=64,
        bm8_mode='candidate' if args.profile=='bm8' else None,stage='official',official=True,adopted=False,
        benchmark_case=case,workload=dict(input_tokens=case[0],output_tokens=1024,concurrency=64,num_prompts=case[3]),
        runner_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),performance_preexcluded_rounds=[1])
    server=evaluator=None
    def save():
        t=root/'state.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(root/'state.json')
        print(json.dumps({k:state.get(k) for k in ('phase','profile_name','benchmark_case','server_pid','evaluator_pid','error')}),flush=True)
    def stop(sig,frame):raise TimeoutError('signal/deadline '+str(sig))
    def terminate(process):
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
    save()
    try:
        remaining=int((DEADLINE-datetime.now(timezone.utc)).total_seconds())
        assert remaining>(7200 if args.case=='16k' else 3600);signal.alarm(remaining)
        command,env,profile,sources=prepare(root,'candidate')
        quality={}
        for batch in (8,64):
            qroot=(Path('/dev/shm/tianshu_r5_BM8_runs_20261024')/f'math{batch}' if args.profile=='bm8'
                else Path(f'/workspace/benchmark-results/20261024_W4_QKVO64_fitted_v2_math{batch}_tianshu'))
            previous=read(qroot/'result.json');q=audit.audit_math(qroot,previous)
            assert previous['phase']=='complete' and previous['accuracy_passed'] and previous['batch']==batch
            assert previous['frozen_sources']==sources and previous['server_command']==command
            assert q==read(qroot/'accuracy_audit.json') and q['above95'] and q['total']==105 and q['all_caps_in_denominator']
            quality[str(batch)]=dict(correct=q['correct'],total=105,accuracy=q['accuracy'],raw_root=str(qroot),
                result_sha256=digest(qroot/'result.json'),accuracy_audit_sha256=digest(qroot/'accuracy_audit.json'))
        if args.case=='4k' and args.profile=='bm8':
            screen=read(EXP/'tianshu_r5_BM8_fresh128_20261024/SCREEN_GATE_32g.json')
            assert screen['complete'] and screen['model_quality_qualified'] and screen['meets_throughput_spread_and_TTFT_gate']
        if args.case=='16k':
            followup=RUNS/('4k_'+args.profile)/'result.json'
            ordered=read(followup) if followup.exists() else read('/workspace/benchmark-results/20261024_W4_QKVO64_fitted_official4k_tianshu/result.json')
            assert ordered['phase']=='complete' and ordered['evaluation_exit_code']==0 and ordered['benchmark_case']==[4096,1024,64,256]
            assert ordered['frozen_sources']==sources,'16K must use the selected version already measured at full4K'
            state['preceding_4K_result_sha256']=digest(followup if followup.exists() else '/workspace/benchmark-results/20261024_W4_QKVO64_fitted_official4k_tianshu/result.json')
        full={**profile,'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']}
        assert len(command)==13 and len(sources)==(1023 if args.profile=='bm8' else 997)
        assert len(full)==(67 if args.profile=='bm8' else 65) and profile['TIANSHU_FUTURE_KV_MODE']=='candidate'
        assert digest(BENCH)=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
        paths=list(BASE.glob('*.py'))+[BASE/'PLAN_32g.md',BENCH,Path('/usr/local/lib/python3.12/site-packages/vllm/benchmarks/serve.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/benchmarks/datasets/datasets.py')]
        controllers={str(v):digest(v) for v in paths}
        state.update(server_command=command,profile_environment=profile,full_environment=full,frozen_sources=sources,
            controller_sources=controllers,quality=quality,deadline=DEADLINE.isoformat(),fresh_API_per_case=True)
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0
        hardware=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_before.txt').write_text(hardware)
        assert 'No running processes found' in hardware
        with (root/'server.log').open('w') as log:server=subprocess.Popen(command,cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='starting_server',server_pid=server.pid);save();limit=time.monotonic()+2400
        while True:
            assert server.poll() is None and time.monotonic()<limit,'API exited or startup timeout'
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response:models=json.load(response)
                if any(v['id']=='minicpm' for v in models['data']):break
            except Exception:pass
            time.sleep(3)
        observed=dict(v.decode().split('=',1) for v in Path(f'/proc/{server.pid}/environ').read_bytes().split(b'\0') if b'=' in v)
        assert {k:observed.get(k) for k in full}==full
        state['observed_commandline']=Path(f'/proc/{server.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        (root/'observed_environment.json').write_text(json.dumps(full,indent=2)+'\n')
        audit.audit_model(root,state)
        def metrics(name):
            with urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=10) as response:(root/name).write_bytes(response.read())
        metrics('before_metrics.txt')
        evaluation=['python3',str(BENCH),'--model','/workspace/MiniCPM5-2B','--served-model-name','minicpm','--port','9031','--test-cases',json.dumps([case],separators=(',',':'))]
        state['evaluation_command']=evaluation;evalenv=dict(env);evalenv.pop('PYTHONPATH',None)
        with (root/'evaluation.log').open('w') as log:evaluator=subprocess.Popen(evaluation,cwd=str(root),env=evalenv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='evaluating',evaluator_pid=evaluator.pid);save()
        while evaluator.poll() is None:
            assert server.poll() is None,'API exited during measurement';time.sleep(5)
        state['evaluation_exit_code']=evaluator.returncode;assert evaluator.returncode==0
        metrics('after_metrics.txt');frozen_check(sources);frozen_check(controllers)
        state['phase']='evaluation_finished_pending_audit'
    except BaseException as error:state.update(phase='failed',error=repr(error))
    finally:
        terminate(evaluator);terminate(server);signal.alarm(0)
        for _ in range(60):
            hardware=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_after.txt').write_text(hardware)
            if 'No running processes found' in hardware:break
            time.sleep(1)
        else:state.update(phase='cleanup_failed',error='GPU not idle')
        save()
    if state['phase']=='evaluation_finished_pending_audit':
        try:
            profile=(audit.audit_model(root,state,runtime=True) if args.case=='4k'
                else audit_long.verify(root,state,audit,r5,read,digest))
            (root/'profile_audit.json').write_text(json.dumps(profile,indent=2)+'\n')
            from verify_statistics import verify
            statistics=verify(root,state,read,digest)
            (root/'statistics_audit.json').write_text(json.dumps(statistics,indent=2)+'\n')
            state.update(phase='complete',statistics=statistics)
        except BaseException as error:state.update(phase='audit_failed',error=repr(error))
    save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
