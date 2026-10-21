# SPDX-License-Identifier: Apache-2.0
"""Exact qualified service, C1 cache salt diagnostic, no observer or new model."""
import argparse,json,os,re,signal,socket,subprocess,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from cache_probe_profile import BASE,CASES,payload,prepare,read,digest,frozen_check

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
    root=args.output.resolve();assert root.is_relative_to(Path('/workspace/benchmark-results')) and not root.exists()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200
    root.mkdir();server=None
    state=dict(phase='preparing',runner_pid=os.getpid(),output_root=str(root),cases=[],diagnostic_only=True,official_benchmark=False,adopted=False,stop_at_utc=args.stop_at_utc)
    def save():
        tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json');print(json.dumps({k:state.get(k) for k in ('phase','server_pid','error')}),flush=True)
    class Stop(BaseException):pass
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);raise Stop(str(signum))
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(s,stop)
    signal.alarm(min(1800,remaining))
    def metrics(name):
        with urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=10) as response:text=response.read().decode()
        (root/name).write_text(text)
        result={}
        for key in ('prefix_cache_queries_total','prefix_cache_hits_total','num_preemptions_total'):
            matches=re.findall(r'^vllm:'+key+r'(?:\{[^}]*\})? ([0-9.eE+\-]+)$',text,re.M);assert len(matches)==1;result[key]=float(matches[0])
        return result
    try:
        command,environment,profile,sources=prepare(root)
        controllers={str(p):digest(p) for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run_probe.sh',BASE/'CPU_GATE_32g.json']}
        frozen_check(controllers)
        for p,h in controllers.items():
            dest=root/'controller_source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes());assert digest(dest)==h
        with socket.socket() as check:assert check.connect_ex(('127.0.0.1',9031))!=0
        hw=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_before.txt').write_text(hw);assert 'No running processes found' in hw
        state.update(server_command=command,profile_environment=profile,frozen_sources=sources,controller_sources=controllers)
        with (root/'server.log').open('w') as log:server=subprocess.Popen(command,cwd='/workspace',env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='starting_server',server_pid=server.pid);save();started=time.monotonic()
        while True:
            assert server.poll() is None and time.monotonic()-started<1200
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response:model=json.load(response)
                if any(r['id']=='minicpm' for r in model['data']):break
            except Exception:pass
            time.sleep(3)
        observed=dict(item.decode().split('=',1) for item in Path(f'/proc/{server.pid}/environ').read_bytes().split(b'\0') if b'=' in item)
        assert {k:observed.get(k) for k in profile}==profile
        (root/'observed_server_environment.json').write_text(json.dumps(dict(environment={k:observed[k] for k in profile},command=command),indent=2)+'\n')
        state['phase']='cache_cases';save()
        for index,(label,salt) in enumerate(CASES):
            before=metrics(f'{index:02d}_{label}_before_metrics.txt');body=payload(salt)
            request=urllib.request.Request('http://127.0.0.1:9031/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST')
            begin=time.monotonic()
            with urllib.request.urlopen(request,timeout=120) as response:answer=json.load(response)
            duration=time.monotonic()-begin;assert answer['usage']==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144)
            choice=answer['choices'][0];assert choice['text'] and len(choice['token_ids'])==16 and all(isinstance(t,int) for t in choice['token_ids'])
            after=metrics(f'{index:02d}_{label}_after_metrics.txt')
            assert before['num_preemptions_total']==after['num_preemptions_total']==0
            state['cases'].append(dict(label=label,cache_salt=salt,payload=body,response=answer,duration_seconds=duration,before_metrics=before,after_metrics=after));save()
        frozen_check(sources);frozen_check(controllers)
        state.update(phase='finished',complete=True)
    except BaseException as e:
        import traceback
        state.update(phase='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
        signal.alarm(0);(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True));save()
    (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='finished' else 1
if __name__=='__main__':raise SystemExit(main())
