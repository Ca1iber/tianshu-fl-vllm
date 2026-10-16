# SPDX-License-Identifier: Apache-2.0
"""Current original-command C64 stage intervals without IXsys instrumentation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_original_command_20261016'))
from validation_profile import prepare,read,digest,frozen_check
from audit_profile import audit_runtime


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True)
    args=parser.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True);remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200
    server=None;state=dict(phase='preparing',runner_pid=os.getpid(),output_root=str(root),official_benchmark=False,adopted=False,
        ixsys_used=False,started_utc=datetime.now(timezone.utc).isoformat(),stop_at_utc=args.stop_at_utc,workloads=[])
    def save():
        temporary=root/'state.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(root/'state.json')
        print(json.dumps({k:state.get(k) for k in ('phase','error','server_pid')}),flush=True)
    class StopRequested(BaseException):pass
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if server is not None and server.poll() is None:os.killpg(server.pid,signal.SIGTERM)
        raise StopRequested(str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
    signal.alarm(min(remaining,1800));save()
    def metrics(name):
        with urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=10) as response:(root/name).write_bytes(response.read())
    def request(length,output,input_seed,api_seed=None):
        rng=random.Random(input_seed)
        payload=dict(model='minicpm',prompt=[rng.randrange(512,120000) for _ in range(length)],max_tokens=output,
            temperature=1.,top_p=.95,ignore_eos=True,stream=True,stream_options=dict(include_usage=True))
        if api_seed is not None:payload['seed']=api_seed
        req=urllib.request.Request('http://127.0.0.1:9031/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
        start=time.perf_counter();usage=None;text='';ttft=None
        with urllib.request.urlopen(req,timeout=600) as response:
            for raw in response:
                line=raw.decode().strip()
                if not line.startswith('data: ') or line=='data: [DONE]':continue
                event=json.loads(line[6:]);choices=event.get('choices',[])
                if choices:
                    chunk=choices[0].get('text','');text+=chunk
                    if chunk and ttft is None:ttft=time.perf_counter()-start
                if event.get('usage'):usage=event['usage']
        elapsed=time.perf_counter()-start
        assert usage and usage['prompt_tokens']==length and usage['completion_tokens']==output and usage['total_tokens']==length+output
        assert text and ttft is not None
        return dict(input_tokens=length,output_tokens=output,input_seed=input_seed,api_seed=api_seed,usage=usage,text=text,ttft_seconds=ttft,wall_seconds=elapsed)
    try:
        cap=Path('/workspace/benchmark-results/20261016_graph_stage_process_tree_v2_tianshu')
        cap_state=read(cap/'result.json')
        for path,sha in cap_state['sources'].items():assert digest(path)==digest(cap/Path(path).name)==sha
        precondition=read(cap/'workload/child/result.json');assert precondition['complete'] and precondition['graph_output_exact']
        assert 5<precondition['unprofiled_bulk_to_single_before_ratio']<20 and 5<precondition['unprofiled_bulk_to_single_after_ratio']<20
        state.update(unprofiled_event_scaling_precondition_verified=True,ixsys_timing_calibration_passed=False,
            precondition_root=str(cap),precondition_sha256=digest(cap/'workload/child/result.json'))
        command,env,profile,sources=prepare(root,Path('/workspace/benchmark-results/20261016_combined_screen_service_tianshu'))
        profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_STAGE_PROFILE_PROBE='1',TIANSHU_STAGE_PROFILE_MODE='events',
            TIANSHU_STAGE_PROFILE_RECORD_DIR=str(root/'stage_records'),TIANSHU_STAGE_PROFILE_ARM_FILE=str(root/'profile_armed'))
        assert len(profile)==39;env.update(profile)
        for path in (Path(__file__),BASE/'hooks/sitecustomize.py',BASE/'audit_profile.py',BASE/'probe_process_tree_v2.py',BASE/'run_coverage_v2.py'):
            sources[str(path)]=digest(path)
        frozen_check(sources)
        for path,sha in sources.items():
            dest=root/'source_snapshot'/Path(path).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(path).read_bytes());assert digest(dest)==sha
        state.update(server_command=command,profile_environment=profile,frozen_sources=sources,
            packages={name:importlib.metadata.version(name) for name in ('torch','vllm','flag-gems')})
        try:connection=socket.create_connection(('127.0.0.1',9031),timeout=2)
        except OSError:pass
        else:connection.close();raise RuntimeError('Port9031 occupied')
        hardware=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_before.txt').write_text(hardware);assert 'No running processes found' in hardware
        with (root/'server.log').open('w') as log:server=subprocess.Popen(command,cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='starting_server',server_pid=server.pid);save();start=time.monotonic()
        while True:
            assert server.poll() is None,'Server exited'
            assert time.monotonic()-start<1200,'Startup20minute limit'
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response:models=json.load(response)
                if any(m['id']=='minicpm' for m in models['data']):break
            except Exception:pass
            time.sleep(3)
        observed=dict(v.decode().split('=',1) for v in Path(f'/proc/{server.pid}/environ').read_bytes().split(b'\0') if b'=' in v)
        assert {k:observed.get(k) for k in profile}==profile
        (root/'observed_server_environment.json').write_text(json.dumps(dict(server_pid=server.pid,environment={k:observed[k] for k in profile},command=command),indent=2)+'\n')
        state['canary_before']=request(128,16,2026101688,777)
        metrics('before_metrics.txt')
        for index,label in enumerate(('warmup','measured')):
            if label=='measured':(root/'profile_armed').touch()
            state['phase']=label;save();start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=64) as pool:
                futures=[pool.submit(request,4096,512,20261016000+1000*index+i) for i in range(64)]
                responses=[f.result() for f in futures]
            state['workloads'].append(dict(label=label,input_tokens=4096,output_tokens=512,concurrency=64,
                responses=responses,duration_seconds=time.perf_counter()-start));save()
        files=list((root/'stage_records').glob('stages_*.json'));assert len(files)==1,'No unique40-frame stage record'
        stages=read(files[0]);assert stages['complete'] and len(stages['frames'])==40 and stages['ixsys_not_used']
        assert stages['hook_source_sha256']==sources[str(BASE/'hooks/sitecustomize.py')]
        for path,metadata in stages['extra_sources'].items():
            assert Path(path).is_relative_to('/usr/local/lib/python3.12/site-packages')
            snapshot=root/'stage_records'/metadata['snapshot'];assert digest(snapshot)==digest(path)==metadata['sha256']
            dest=root/'source_snapshot'/Path(path).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(snapshot.read_bytes());sources[path]=metadata['sha256']
        state['frozen_sources']=sources
        state['canary_after']=request(128,16,2026101688,777)
        assert state['canary_before']['text']==state['canary_after']['text']
        metrics('after_metrics.txt');frozen_check(sources)
        state.update(phase='finished_pending_audit',stage_record=str(files[0]))
    except StopRequested as error:state.update(phase='stopped',error=str(error))
    except Exception as error:state.update(phase='failed',error=repr(error))
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
        signal.alarm(0);(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True));save()
    if state['phase']=='finished_pending_audit':
        try:
            state['runtime_audit']=audit_runtime(root,state,official=True)
            frames=stages['frames'];gpu={key:[r['gpu_stream_intervals_ms'][key] for r in frames] for key in ('total','forward','logits','sample')}
            host={key:{kind:statistics.median(r['cpu_call_spans'][key][kind] for r in frames) for kind in ('wall_ms','thread_cpu_ms')}
                for key in ('execute','sample_tokens','forward','logits','sample')}
            sums={key:sum(values) for key,values in gpu.items()};medians={key:statistics.median(values) for key,values in gpu.items()}
            summary=dict(complete=True,official_benchmark=False,frame_count=40,ixsys_used=False,
                gpu_stream_interval_sum_ms=sums,gpu_stream_interval_median_ms=medians,
                fractions_of_total_stream_interval={key:sums[key]/sums['total'] for key in ('forward','logits','sample')},
                remaining_stream_interval_ms=sums['total']-sums['forward']-sums['logits']-sums['sample'],
                host_call_median=host,seeded_canary_before_after_equal=True,
                scope='Stream intervals include submission gaps. Host wall spans include blocked waits. Thread CPU excludes blocking. Not individual kernel active time.')
            (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');state.update(phase='complete',complete=True,summary=summary)
        except Exception as error:state.update(phase='audit_failed',error=repr(error))
    save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    if state['phase']=='complete':subprocess.run([sys.executable,str(BASE.parent/'archive_completed_pair.py'),
        '--single-probe','--source',str(root),'--label','model_stage_events_20261016'],check=True)
    return 0 if state['phase']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
