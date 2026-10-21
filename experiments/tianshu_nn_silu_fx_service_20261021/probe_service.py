# SPDX-License-Identifier: Apache-2.0
"""Short actual-model fusion integration, no benchmark or model-quality claim."""
import argparse,json,os,re,signal,socket,subprocess,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from candidate_profile import BASE,payload,prepare,digest,frozen_check

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('control','candidate'),required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1800
    server=None;state=dict(phase='preparing',mode=args.mode,diagnostic_only=True,benchmark=False,adopted=False,model_quality_not_evaluated=True,requests=[],runner_pid=os.getpid(),stop_at_utc=args.stop_at_utc)
    def save():
        temp=root/'state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'state.json');print(json.dumps({k:state.get(k) for k in ('phase','server_pid','error')}),flush=True)
    class Stop(BaseException):pass
    def stop(signum,frame):signal.signal(signal.SIGTERM,signal.SIG_IGN);raise Stop(str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
    signal.alarm(min(remaining,1800))
    def metrics(name):
        with urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=10) as response:text=response.read().decode()
        (root/name).write_text(text);return text
    def request(label,index=0,output=16):
        body=payload('NN-silu-FX-'+args.mode+'-'+label,index,output)
        req=urllib.request.Request('http://127.0.0.1:9031/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST');start=time.perf_counter()
        with urllib.request.urlopen(req,timeout=240) as response:answer=json.load(response)
        row=dict(label=label,payload=body,response=answer,duration_seconds=time.perf_counter()-start)
        (root/(label+'_response.json')).write_text(json.dumps(row,indent=2)+'\n')
        assert {k:answer['usage'].get(k) for k in ('prompt_tokens','completion_tokens','total_tokens')}==dict(prompt_tokens=128,completion_tokens=output,total_tokens=128+output)
        assert len(answer['choices'][0]['token_ids'])==output
        return row
    try:
        command,environment,profile,sources=prepare(root,args.mode)
        controllers={str(q):digest(q) for q in [*BASE.rglob('*.py'),BASE/'PLAN_32g.md',BASE/'run_smoke.sh',BASE/'CPU_GATE_32g.json']};frozen_check(controllers)
        state.update(server_command=command,profile_environment=profile,frozen_sources=sources,controller_sources=controllers)
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0
        hw=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_before.txt').write_text(hw);assert 'No running processes found' in hw
        with (root/'server.log').open('w') as log:server=subprocess.Popen(command,cwd='/workspace',env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='starting_server',server_pid=server.pid);save();started=time.monotonic()
        while True:
            assert server.poll() is None and time.monotonic()-started<1200
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response:models=json.load(response)
                if any(r['id']=='minicpm' for r in models['data']):break
            except Exception:pass
            time.sleep(3)
        observed=dict(v.decode().split('=',1) for v in Path(f'/proc/{server.pid}/environ').read_bytes().split(b'\0') if b'=' in v)
        assert {k:observed.get(k) for k in profile}==profile
        (root/'observed_server_environment.json').write_text(json.dumps(dict(environment={k:observed[k] for k in profile},command=command),indent=2)+'\n')
        state['phase']='cold_C1';save();metrics('before_metrics.txt')
        state['requests']=[request('cold_A'),request('cold_B')];save()
        assert state['requests'][0]['response']['choices'][0]['token_ids']==state['requests'][1]['response']['choices'][0]['token_ids']
        state['phase']='C64_wave';save();metrics('wave_before_metrics.txt')
        with ThreadPoolExecutor(max_workers=64) as pool:state['wave']=list(pool.map(lambda i:request(f'wave_{i:02d}',100+i,128),range(64)))
        metrics('wave_after_metrics.txt');save()
        state['requests'].append(request('cold_C_after'));save()
        assert state['requests'][0]['response']['choices'][0]['token_ids']==state['requests'][-1]['response']['choices'][0]['token_ids']
        metrics('after_metrics.txt');frozen_check(sources);frozen_check(controllers)
        log=(root/'server.log').read_text(errors='replace');assert log.count('Application startup complete.')==1
        assert 'Triton autotuning for function' not in log.split('Application startup complete.',1)[1]
        state.update(phase='finished',complete=True)
    except BaseException as error:
        import traceback
        state.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
        signal.alarm(0);save()
    (root/'result.json').write_text(json.dumps(state,indent=2)+'\n');return 0 if state['phase']=='finished' else 1
if __name__=='__main__':raise SystemExit(main())
