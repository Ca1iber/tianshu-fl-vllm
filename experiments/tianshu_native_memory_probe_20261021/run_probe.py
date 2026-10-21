# SPDX-License-Identifier: Apache-2.0
"""Startup+eight C8 canaries, native memory observations; no throughput benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import random,signal,socket,subprocess,time,urllib.request
from candidate_profile import BASE,prepare,read,digest,frozen_check

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('control','candidate'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    seconds=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert seconds>600
    server=None
    def stop(signum,frame):
        if server is not None and server.poll() is None:os.killpg(server.pid,signal.SIGKILL)
        raise TimeoutError("Native memory observation reached stop signal "+str(signum))
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    signal.alarm(seconds)
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    root.mkdir();(root/'hardware_before.txt').write_text(hardware)
    state=dict(complete=False,phase='preparing',mode=args.mode,runner_pid=os.getpid(),benchmark=False,model_Math=False,KV_budget_not_modified=True)
    server=None;failure=None
    def save():
        tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json')
    save()
    try:
        command,environment,profile,sources=prepare(root,args.mode);cpu=read(BASE/'CPU_GATE_32g.json')
        assert sources==next(r['sources'] for r in cpu['profiles'] if r['mode']==args.mode)
        state.update(server_command=command,profile_environment=profile,frozen_sources=sources,CPU_gate_sha256=digest(BASE/'CPU_GATE_32g.json'))
        try:connection=socket.create_connection(('127.0.0.1',9031),timeout=1)
        except OSError:pass
        else:connection.close();raise RuntimeError('Port9031 occupied')
        with (root/'server.log').open('w') as log:server=subprocess.Popen(command,cwd='/workspace',env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='starting_server',server_pid=server.pid);save()
        until=time.monotonic()+min(seconds-60,1800)
        while True:
            assert server.poll() is None,'Server exited during startup'
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response:ready=json.load(response)
                if any(r['id']=='minicpm' for r in ready['data']):break
            except Exception:pass
            assert time.monotonic()<until,'Startup observation deadline';time.sleep(2)
        raw=Path(f'/proc/{server.pid}/environ').read_bytes();observed=dict(item.decode().split('=',1) for item in raw.split(b'\0') if item)
        assert {key:observed.get(key) for key in profile}==profile
        state.update(phase='C8_canaries',selected_actual_environment={key:observed[key] for key in profile});save()
        def request(i):
            rng=random.Random(8971+i)
            payload=dict(model='minicpm',prompt=[rng.randrange(512,120000) for _ in range(128)],max_tokens=16,ignore_eos=True,temperature=1.,top_p=.95,seed=777,cache_salt='native-memory-observe-'+str(i),stream=False)
            req=urllib.request.Request('http://127.0.0.1:9031/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)
        with ThreadPoolExecutor(max_workers=8) as pool:responses=list(pool.map(request,range(8)))
        (root/'canary_responses.json').write_text(json.dumps(responses,indent=2)+'\n')
        assert all(r['usage']==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144) for r in responses)
        files=list((root/'native_memory_probe_records').glob('observed_*.json'));assert len(files)==1;record=read(files[0]);assert record['complete'] and record['mode']==args.mode
        assert record['observer_sha256']==sources[record['observer_source']]
        log=(root/'server.log').read_text(errors='replace');assert log.count('Application startup complete.')==1
        configs=[r for file in (root/'static_compile_records').glob('config_*.json') for r in read(file)]
        assert configs and all(r['actual_compile_sizes']==([] if args.mode=='control' else [8,64]) for r in configs)
        state.update(phase='observed',native_observation=record,actual_native_configs=configs,canary_count=8,
            canaries=[dict(usage=r['usage'],text_sha256=hashlib.sha256(r['choices'][0]['text'].encode()).hexdigest()) for r in responses],
            full_C64_replay_not_tested=True,model_speedup_not_claimed=True,quality_not_claimed=True)
        frozen_check(sources);save()
    except BaseException as exc:
        failure=exc;state.update(phase='failed',error=repr(exc));save()
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)
        state.update(server_reaped=(server is None or server.poll() is not None))
        if failure is None:state.update(phase='complete',complete=True)
        save()
    if failure is not None:raise failure
    print(json.dumps({key:state[key] for key in ('phase','complete','native_observation','canary_count','server_reaped')}))
if __name__=='__main__':main()
