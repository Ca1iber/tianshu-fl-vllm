# SPDX-License-Identifier: Apache-2.0
"""Current qualified profile: outside-Graph events, diagnostic workload only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
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
from stage_measurement_profile import prepare, audit_runtime, digest, frozen_check
BASE = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stop-at-utc', required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.is_relative_to(Path('/workspace/benchmark-results')) and not root.exists()
    remaining = int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1200
    root.mkdir(parents=True)
    server = None
    state = dict(phase='preparing', runner_pid=os.getpid(), output_root=str(root), mode='candidate', actual_budget=4096,
                 official_benchmark=False, adopted=False, started_utc=datetime.now(timezone.utc).isoformat(),
                 stop_at_utc=args.stop_at_utc, workloads=[])
    def save():
        tmp = root/'state.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(root/'state.json')
        print(json.dumps({k: state.get(k) for k in ('phase','error','server_pid')}), flush=True)
    class StopRequested(BaseException):
        pass
    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise StopRequested(str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.alarm(min(2400, remaining))
    def metrics(name):
        with urllib.request.urlopen('http://127.0.0.1:9031/metrics', timeout=10) as response:
            (root/name).write_bytes(response.read())
    def request(length, output, input_seed, api_seed=None):
        rng = random.Random(input_seed)
        payload = dict(model='minicpm', prompt=[rng.randrange(512,120000) for _ in range(length)],
                       max_tokens=output, temperature=1., top_p=.95, ignore_eos=True, stream=True,
                       stream_options=dict(include_usage=True))
        if api_seed is not None:
            payload['seed'] = api_seed
        req = urllib.request.Request('http://127.0.0.1:9031/v1/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type':'application/json'}, method='POST')
        start = time.perf_counter()
        usage, text, ttft = None, '', None
        with urllib.request.urlopen(req, timeout=600) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data: ') or line == 'data: [DONE]':
                    continue
                event = json.loads(line[6:])
                if event.get('choices'):
                    chunk = event['choices'][0].get('text','')
                    text += chunk
                    if chunk and ttft is None:
                        ttft = time.perf_counter()-start
                if event.get('usage'):
                    usage = event['usage']
        assert usage == dict(prompt_tokens=length, completion_tokens=output, total_tokens=length+output), usage
        assert text and ttft is not None
        return dict(input_tokens=length, output_tokens=output, input_seed=input_seed, api_seed=api_seed,
                    usage=usage, text=text, ttft_seconds=ttft, wall_seconds=time.perf_counter()-start)
    try:
        command, environment, profile, sources = prepare(root)
        assert len(profile) == 62
        frozen_check(sources)
        for path,sha in sources.items():
            dest = root/'source_snapshot'/Path(path).relative_to('/')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(path).read_bytes())
            assert digest(dest) == sha
        state.update(server_command=command, profile_environment=profile, frozen_sources=sources)
        with socket.socket() as check:
            assert check.connect_ex(('127.0.0.1',9031)) != 0, 'Port9031 occupied'
        hardware = subprocess.check_output(['ixsmi'], text=True)
        (root/'hardware_before.txt').write_text(hardware)
        assert 'No running processes found' in hardware
        with (root/'server.log').open('w') as log:
            server = subprocess.Popen(command,cwd='/workspace',env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='starting_server', server_pid=server.pid)
        save()
        start = time.monotonic()
        while True:
            assert server.poll() is None, 'Server exited'
            assert time.monotonic()-start < 1200, 'Startup20minute limit'
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response:
                    models = json.load(response)
                if any(m['id']=='minicpm' for m in models['data']):
                    break
            except Exception:
                pass
            time.sleep(3)
        observed = dict(v.decode().split('=',1) for v in Path(f'/proc/{server.pid}/environ').read_bytes().split(b'\0') if b'=' in v)
        assert {k:observed.get(k) for k in profile} == profile
        (root/'observed_server_environment.json').write_text(json.dumps(dict(server_pid=server.pid,environment={k:observed[k] for k in profile},command=command),indent=2)+'\n')
        state['canary_before'] = request(128,16,20261021088,777)
        metrics('before_metrics.txt')
        for index,label in enumerate(('warmup','measured')):
            if label == 'measured':
                (root/'profile_armed').touch()
            state['phase'] = label
            save()
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=64) as pool:
                responses = list(pool.map(lambda i:request(4096,512,202610210000+index*1000+i),range(64)))
            state['workloads'].append(dict(label=label,input_tokens=4096,output_tokens=512,concurrency=64,
                                           responses=responses,duration_seconds=time.perf_counter()-start))
            save()
        files = list((root/'stage_records').glob('stages_*.json'))
        assert len(files) == 1, 'No unique40-frame stage record'
        stages = read(files[0])
        assert stages['complete'] and len(stages['frames']) == 40 and stages['ixsys_not_used']
        assert stages['hook_source_sha256'] == sources[str(BASE/'hooks/sitecustomize.py')]
        for path,metadata in stages['extra_sources'].items():
            snap = root/'stage_records'/metadata['snapshot']
            assert digest(snap) == digest(path) == metadata['sha256']
            dest = root/'source_snapshot'/Path(path).relative_to('/')
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(snap.read_bytes())
            sources[path] = metadata['sha256']
        state['canary_after'] = request(128,16,20261021088,777)
        assert state['canary_before']['text'] == state['canary_after']['text']
        metrics('after_metrics.txt')
        frozen_check(sources)
        state.update(phase='finished_pending_audit',stage_record=str(files[0]),frozen_sources=sources)
    except BaseException as error:
        state.update(phase='failed',error=repr(error))
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL)
                server.wait(timeout=5)
        signal.alarm(0)
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
        save()
    if state['phase'] == 'finished_pending_audit':
        try:
            state['runtime_audit'] = audit_runtime(root,state,official=True)
            frames = stages['frames']
            keys = ('total','forward','logits','sample')
            medians = {k:statistics.median(r['gpu_stream_intervals_ms'][k] for r in frames) for k in keys}
            sums = {k:sum(r['gpu_stream_intervals_ms'][k] for r in frames) for k in keys}
            summary = dict(complete=True,official_benchmark=False,frame_count=40,ixsys_used=False,
                           gpu_stream_interval_sum_ms=sums,gpu_stream_interval_median_ms=medians,
                           fractions_of_total_stream_interval={k:sums[k]/sums['total'] for k in keys[1:]},
                           host_call_median={k:{kind:statistics.median(r['cpu_call_spans'][k][kind] for r in frames)
                               for kind in ('wall_ms','thread_cpu_ms')} for k in ('execute','sample_tokens','forward','logits','sample')},
                           seeded_canary_before_after_equal=True,
                           scope='Outside-Graph stream intervals include host submission gaps, not individual kernel active time.')
            (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            state.update(phase='complete',complete=True,summary=summary)
        except Exception as error:
            state.update(phase='audit_failed',error=repr(error))
    save()
    (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' else 1

if __name__ == '__main__':
    raise SystemExit(main())
