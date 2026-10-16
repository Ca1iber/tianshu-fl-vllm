# SPDX-License-Identifier: Apache-2.0
"""Original serve argv, N64 different-input-seed screen, RNG-only A/B change."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'tianshu_original_command_20261016'))
from validation_profile import prepare, digest, read, frozen_check
from audit_profile import audit_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('control', 'candidate'), required=True)
    parser.add_argument('--micro-root', type=Path, required=True)
    parser.add_argument('--stop-at-utc', required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True)
    stop_at = datetime.fromisoformat(args.stop_at_utc)
    remaining = int((stop_at-datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1200
    state = dict(phase='preparing', mode=args.mode, output_root=str(root), runner_pid=os.getpid(),
                 original_serve_command=True, official_benchmark=False, adopted=False,
                 short_4k_rounds=[], seeded_canaries=[], stop_at_utc=stop_at.isoformat(),
                 workload=dict(input_tokens=4096, output_tokens=1024, concurrency=64,
                               requests_per_round=64, rounds=3, excluded_rounds=[1]))
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
            os.killpg(server.pid, signal.SIGTERM)
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
        micro=read(args.micro_root/'result.json')
        assert micro['complete'] and len(micro['cases'])==105 and len(micro['rng_words'])==12
        assert all(c['cpu_reference_equal'] and c['production_same_noise_equal'] and c['materialized_matches_fused'] for c in micro['cases'])
        assert all(c['exact'] for c in micro['rng_words'])
        assert all(c['exact_advance'] and c['reset_reproducible'] and c['successive_draws_differ'] for c in micro['state_checks'])
        assert all(c['passed'] for c in micro['distribution_checks'])
        command, env, profile, sources=prepare(root, Path('/workspace/benchmark-results/20261016_combined_screen_service_tianshu'))
        profile.update(PYTHONPATH=str(BASE/'hooks'), TIANSHU_RNG_SAMPLER_PROBE='1' if args.mode=='candidate' else '0',
                       TIANSHU_RNG_SAMPLER_RECORD_DIR=str(root/'rng_sampler_records'))
        assert len(profile)==37
        env.update(profile)
        for p,h in micro['sources'].items(): assert digest(p)==h; sources[p]=h
        for path in (Path(__file__), BASE/'audit_profile.py', BASE/'run_service_pair.py', BASE/'hooks/sitecustomize.py'):
            sources[str(path)]=digest(path)
        frozen_check(sources)
        for path,h in sources.items():
            dest=root/'source_snapshot'/Path(path).relative_to('/')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(path).read_bytes());assert digest(dest)==h
        state.update(server_command=command, profile_environment=profile, frozen_sources=sources,
                     started_at_utc=datetime.now(timezone.utc).isoformat(), micro_root=str(args.micro_root))
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
        for index in range(3):
            state['phase']=f'round_{index+1}';save()
            log=(root/'server.log').read_text(errors='replace');boundary=len(log)
            metrics(f'round{index+1}_before_metrics.txt');start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=64) as pool:
                futures=[pool.submit(request,4096,1024,2026101600+100*index+i) for i in range(64)]
                responses=[f.result() for f in futures]
            duration=time.perf_counter()-start;metrics(f'round{index+1}_after_metrics.txt')
            tuning=[l for l in (root/'server.log').read_text(errors='replace')[boundary:].splitlines() if 'Triton autotuning for function' in l]
            state['short_4k_rounds'].append(dict(round=index+1,duration_seconds=duration,total_tokens_per_second=64*5120/duration,
                output_tokens_per_second=64*1024/duration,mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in responses),
                successful_requests=64,responses=responses,tuning_records=tuning));save()
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
            state['profile_audit']=audit_runtime(root,state,official=True)
            records=[read(p) for p in (root/'rng_sampler_records').glob('sampler_*.json')]
            if args.mode=='control':assert not records
            else:
                assert any(r['route']=='fused' and r['batch']==64 and r['seeded_rows']==0 for r in records)
                assert any(r['route']=='seeded_fallback' and r['seeded_rows']==1 for r in records)
                assert all(r['kernel_sha256']==micro['sources'][str(BASE/'experimental_rng_sampler_v2.py')] for r in records)
            values=[r['total_tokens_per_second'] for r in state['short_4k_rounds'][1:]]
            state.update(phase='complete',mean_total_tok_s=statistics.mean(values),spread_percent=100*(max(values)-min(values))/statistics.mean(values))
        except Exception as error:state.update(phase='audit_failed',error=repr(error))
    save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
