# SPDX-License-Identifier: Apache-2.0
"""Original serve argv with qualified operators and software8 KV A/B."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import importlib.util
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

BASE = Path(__file__).resolve().parent
from candidate_profile import prepare, digest, read, frozen_check
spec=importlib.util.spec_from_file_location('_batch_meta_service_audit',BASE/'audit_profile.py')
_audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(_audit)
audit_runtime=_audit.audit_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('control', 'candidate'), required=True)
    parser.add_argument('--stop-at-utc', required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    nreq, ilen, olen, nrounds = (8,512,128,1) if args.smoke else (64,4096,1024,3)
    root = args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True)
    stop_at = datetime.fromisoformat(args.stop_at_utc)
    remaining = int((stop_at-datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1200
    state = dict(phase='preparing', mode=args.mode, output_root=str(root), runner_pid=os.getpid(),
                 original_serve_command=True, official_benchmark=False, adopted=False,
                 short_4k_rounds=[], seeded_canaries=[], stop_at_utc=stop_at.isoformat(),
                 workload=dict(input_tokens=ilen, output_tokens=olen, concurrency=nreq,
                               requests_per_round=nreq, rounds=nrounds, excluded_rounds=[] if args.smoke else [1]))
    server = None

    def save():
        tmp = root/'state.tmp'; tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(root/'state.json')
        print(json.dumps(dict(phase=state['phase'], mode=args.mode, rounds=len(state['short_4k_rounds']), error=state.get('error'))), flush=True)

    class StopRequested(BaseException):
        pass

    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGKILL)
        raise StopRequested(str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM): signal.signal(sig, stop)
    signal.alarm(remaining)

    def metrics(name):
        with urllib.request.urlopen('http://127.0.0.1:9031/metrics', timeout=10) as response:
            (root/name).write_bytes(response.read())

    def request(length, output, input_seed, api_seed=None):
        rng = random.Random(input_seed)
        payload = dict(model='minicpm', prompt=[rng.randrange(512, 120000) for _ in range(length)],
                       max_tokens=output, ignore_eos=True, temperature=1.0, top_p=.95,
                       repetition_penalty=1.0, logprobs=None, stream=True,
                       stream_options=dict(include_usage=True))
        if api_seed is not None: payload['seed']=api_seed
        req = urllib.request.Request('http://127.0.0.1:9031/v1/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
        started=time.perf_counter(); text=''; ttft=None; usage=None
        with urllib.request.urlopen(req, timeout=900) as response:
            for raw in response:
                line=raw.decode().strip()
                if not line.startswith('data: ') or line == 'data: [DONE]': continue
                event=json.loads(line[6:]); choices=event.get('choices', [])
                if choices:
                    chunk=choices[0].get('text',''); text+=chunk
                    if chunk and ttft is None: ttft=time.perf_counter()-started
                if event.get('usage'): usage=event['usage']
        duration=time.perf_counter()-started
        assert usage and usage['prompt_tokens']==length and usage['completion_tokens']==output
        assert usage['total_tokens']==length+output and ttft is not None
        return dict(input_seed=input_seed, api_seed=api_seed, usage=usage, text=text,
                    ttft_seconds=ttft, wall_seconds=duration, input_tokens=length, output_tokens=output)

    save()
    try:
        prerequisite=read(BASE/'GPU_GATE_32g.json')
        assert prerequisite['complete'] and prerequisite['actual_builder_writer_forward_cases']==21
        assert read(Path(prerequisite['remote_result']))['complete']
        assert digest(Path(prerequisite['remote_result']))==prerequisite['remote_result_sha256']
        frozen_check(prerequisite['frozen_sources'])
        command, env, profile, sources=prepare(root,args.mode)
        sources[str(BASE/'GPU_GATE_32g.json')]=digest(BASE/'GPU_GATE_32g.json')
        gate_dest=root/'source_snapshot'/BASE.relative_to('/')/'GPU_GATE_32g.json'
        gate_dest.parent.mkdir(parents=True,exist_ok=True);gate_dest.write_bytes((BASE/'GPU_GATE_32g.json').read_bytes())
        state.update(server_command=command, profile_environment=profile, frozen_sources=sources,
                     started_at_utc=datetime.now(timezone.utc).isoformat())
        try: connection=socket.create_connection(('127.0.0.1',9031),timeout=2)
        except OSError: pass
        else: connection.close(); raise RuntimeError('Port9031 occupied')
        hardware=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_before.txt').write_text(hardware)
        assert 'No running processes found' in hardware
        with (root/'server.log').open('w') as log:
            server=subprocess.Popen(command,cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='starting_server',server_pid=server.pid);save()
        started=time.monotonic()
        while True:
            assert server.poll() is None,'Server exited during startup'
            assert time.monotonic()-started<1800,'Startup exceeded30minutes'
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response: models=json.load(response)
                if any(m['id']=='minicpm' for m in models['data']):break
            except Exception:pass
            time.sleep(3)
        observed=dict(v.decode().split('=',1) for v in Path(f'/proc/{server.pid}/environ').read_bytes().split(b'\0') if b'=' in v)
        assert {k:observed.get(k) for k in profile}==profile
        (root/'observed_server_environment.json').write_text(json.dumps(dict(server_pid=server.pid,environment={k:observed[k] for k in profile},command=command),indent=2)+'\n')
        state['phase']='seeded_canaries';save()
        for _ in range(2):state['seeded_canaries'].append(request(128,16,2026101699,777))
        assert state['seeded_canaries'][0]['text']==state['seeded_canaries'][1]['text']
        metrics('before_metrics.txt')
        for index in range(nrounds):
            state['phase']=f'round_{index+1}';save()
            log=(root/'server.log').read_text(errors='replace');boundary=len(log)
            metrics(f'round{index+1}_before_metrics.txt');start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=nreq) as pool:
                futures=[pool.submit(request,ilen,olen,202610193000+100*index+i) for i in range(nreq)]
                responses=[f.result() for f in futures]
            duration=time.perf_counter()-start;metrics(f'round{index+1}_after_metrics.txt')
            tuning=[l for l in (root/'server.log').read_text(errors='replace')[boundary:].splitlines() if 'Triton autotuning for function' in l]
            state['short_4k_rounds'].append(dict(round=index+1,duration_seconds=duration,total_tokens_per_second=nreq*(ilen+olen)/duration,
                output_tokens_per_second=nreq*olen/duration,mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in responses),
                successful_requests=nreq,responses=responses,tuning_records=tuning));save()
        metrics('after_metrics.txt');frozen_check(sources)
        state['phase']='finished_pending_audit'
    except StopRequested as error:state.update(phase='stopped',error=str(error))
    except Exception as error:state.update(phase='failed',error=repr(error))
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
        signal.alarm(0)
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
        save()
    if state['phase']=='finished_pending_audit':
        try:
            state['profile_audit']=audit_runtime(root,state,official=not args.smoke)
            values=[r['total_tokens_per_second'] for r in state['short_4k_rounds'][0 if args.smoke else 1:]]
            state.update(phase='complete',mean_total_tok_s=statistics.mean(values),spread_percent=100*(max(values)-min(values))/statistics.mean(values))
        except Exception as error:state.update(phase='audit_failed',error=repr(error))
    save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    if args.smoke and state['phase']=='complete':
        subprocess.run([sys.executable,str(BASE.parent/'archive_completed_pair.py'),'--single-probe','--source',str(root),'--label','soft8_engine_v2_smoke_20261019'],check=True)
    return 0 if state['phase']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
