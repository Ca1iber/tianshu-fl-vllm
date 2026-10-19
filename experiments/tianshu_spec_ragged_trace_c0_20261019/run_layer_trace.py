# SPDX-License-Identifier: Apache-2.0
"""Real model numerical gate for bounded mixed-query graphs; not a performance score."""
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
RAGGED=BASE.parent/'tianshu_spec_ragged_graph_20261019'
sys.path.insert(0,str(RAGGED))
sys.path.insert(0, str(BASE.parent / 'tianshu_gemm_topp_geo_20261016'))
from validation_profile_new import prepare, digest, read, frozen_check
spec=importlib.util.spec_from_file_location('_batch_meta_service_audit',RAGGED/'audit_probe.py')
_audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(_audit)
audit_runtime=_audit.audit_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('candidate',), required=True)
    parser.add_argument('--stop-at-utc', required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True)
    stop_at = datetime.fromisoformat(args.stop_at_utc)
    remaining = int((stop_at-datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1200
    state = dict(phase='preparing', mode=args.mode, output_root=str(root), runner_pid=os.getpid(),
                 original_serve_arguments_preserved=True, extra_speculative_config=True, official_benchmark=False, adopted=False,
                 short_4k_rounds=[], seeded_canaries=[], flush_canaries=[], diagnostic_ragged_model_numeric=True, layer_diagnostic=True, individual_completed_responses_retained=True, stop_at_utc=stop_at.isoformat(),
                 workload=dict(input_tokens=4096, output_tokens=512, concurrency=64,
                               requests_per_round=64, rounds=2, excluded_rounds=[]))
    server = None

    def save():
        tmp = root/'state.tmp'; tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(root/'state.json')
        print(json.dumps(dict(phase=state['phase'], mode=args.mode, rounds=len(state['short_4k_rounds']), error=state.get('error'))), flush=True)

    def phase_marker(value):
        destination=root/'coverage_phase.txt'
        tmp=destination.with_suffix('.tmp')
        tmp.write_text(value)
        tmp.replace(destination)

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
        row=dict(input_seed=input_seed, api_seed=api_seed, usage=usage, text=text,
                    ttft_seconds=ttft, wall_seconds=duration, input_tokens=length, output_tokens=output)
        responses=root/'individual_responses';responses.mkdir(exist_ok=True)
        path=responses/f'{time.time_ns()}_{input_seed}_{api_seed}.json'
        with path.open('x') as out:json.dump(row,out)
        return row

    save()
    try:
        cpu_gate=read(RAGGED/'CPU_POLICY_AUDIT_32g.json')
        assert cpu_gate['complete'] and cpu_gate['generated_metadata_cases']==42 and cpu_gate['GPU_test'] is False
        assert cpu_gate['mixed_routes']+cpu_gate['native_fallback_cases']==2144
        for relative,expected in cpu_gate['sources'].items():assert digest(RAGGED/relative)==expected
        FIXED=BASE.parent/'tianshu_ngram_fixed_index_20261016'
        fixed_gate=read(FIXED/'INDEX_MICRO_GATE_32g.json')
        assert fixed_gate['complete'] and fixed_gate['all48_public_API_checks'] and fixed_gate['all6_Torch_API_routes']
        fixed_path=Path(fixed_gate['remote_result']);assert digest(fixed_path)==fixed_gate['remote_result_sha256']
        for path,sha in read(fixed_path)['source_snapshots'].items():assert digest(Path(path))==sha
        NGRAM=BASE.parent/'tianshu_ngram_framework_20261016'
        prerequisite=read(NGRAM/'NG_MICRO_GATE_32g.json')
        assert prerequisite['complete'] and prerequisite['all48_rejection_cases'] and prerequisite['all14_attention_phases']
        for name,entry in prerequisite['remote_results'].items():
            path=Path(entry['path']);assert digest(path)==entry['sha256']
            raw=read(path)
            for source,sha in raw['source_snapshots'].items():assert digest(Path(source))==sha
        command, env, profile, sources=prepare(root, Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
        reference=read(Path('/workspace/benchmark-results/20261016_request_positions_service_tianshu/control/result.json'))
        assert reference['phase']=='complete' and reference['server_command']==command
        for source,sha in sources.items():assert reference['frozen_sources'][source]==sha
        speculative=dict(method='ngram',num_speculative_tokens=1,prompt_lookup_min=3,prompt_lookup_max=5)
        command=command+['--speculative-config',json.dumps(speculative)]
        config_index=command.index('--compilation-config')+1
        compilation=json.loads(command[config_index]);compilation['mode']=0
        command[config_index]=json.dumps(compilation)
        state['diagnostic_compile_mode']='NONE; FULL CUDA Graph retained for observable Python module calls'
        state['original_serve_arguments_preserved']=False
        state.update(reference_control_root=reference['output_root'],speculative_config=speculative,
                     reference_scope='Configuration provenance only; diagnostic input seeds differ; throughput not a candidate score')
        profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_NGRAM_MODE=args.mode,
                       TIANSHU_DSPARK_RECORD_DIR=str(root/'ngram_records'))
        profile.update(TIANSHU_RAGGED_RECORD_DIR=str(root/'ragged_records'),
                       TIANSHU_RAGGED_NUMERIC_MODE='1')
        assert len(profile)==42
        env.update(profile)
        relatives=('hooks/sitecustomize.py','run_pilot.py','audit_pilot.py','INDEX_MICRO_GATE_32g.json','probe_index.py')
        extra=[FIXED/path for path in relatives]
        extra.extend(BASE.glob('*.py')); extra.append(BASE/'hooks/sitecustomize.py')
        extra.extend(RAGGED.glob('*.py'));extra.append(RAGGED/'hooks/sitecustomize.py')
        extra.append(RAGGED/'CPU_POLICY_AUDIT_32g.json')
        extra.append(BASE.parent/'tianshu_spec_graph_coverage_20261019/route_observer.py')
        extra.append(Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py'))
        for relative in ('hooks/sitecustomize.py','run_pilot.py','audit_pilot.py','NG_MICRO_GATE_32g.json',
                         'native_verify.py','probe_attention_v2.py','probe_rejection.py'):
            extra.append(NGRAM/relative)
        extra.append(BASE.parent/'tianshu_dspark_20261012/fixed_index.py')
        extra.append(BASE.parent/'tianshu_dspark_20261012/rejection_compat.py')
        installed=Path('/usr/local/lib/python3.12/site-packages')
        for relative in ('flag_gems/ops/index.py','vllm/v1/sample/rejection_sampler.py','vllm/v1/spec_decode/ngram_proposer.py',
                         'vllm/config/speculative.py','vllm/config/model.py','vllm/v1/sample/sampler.py',
                         'vllm/v1/attention/ops/triton_unified_attention.py','vllm/v1/attention/ops/triton_attention_helpers.py',
                         'vllm/v1/attention/backends/triton_attn.py'):
            extra.append(installed/relative)
        extra.extend(installed/p for p in ('vllm_fl/compilation/graph.py','vllm_fl/worker/model_runner.py','vllm/v1/cudagraph_dispatcher.py','vllm/forward_context.py'))
        for path in extra:sources[str(path)]=digest(path)
        frozen_check(sources)
        for path,h in sources.items():
            dest=root/'source_snapshot'/Path(path).relative_to('/')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(path).read_bytes());assert digest(dest)==h
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
        phase_marker('seeded_canaries')
        state['phase']='seeded_canaries';save()
        for _ in range(2):state['seeded_canaries'].append(request(128,16,2026101699,777))
        assert state['seeded_canaries'][0]['text']==state['seeded_canaries'][1]['text']
        metrics('before_metrics.txt')
        for index in range(2):
            state['phase']=f'round_{index+1}';phase_marker(state['phase']);save()
            log=(root/'server.log').read_text(errors='replace');boundary=len(log)
            metrics(f'round{index+1}_before_metrics.txt');start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=64) as pool:
                futures=[pool.submit(request,4096,512,202610191000+100*index+i) for i in range(64)]
                responses=[f.result() for f in futures]
            duration=time.perf_counter()-start;metrics(f'round{index+1}_after_metrics.txt')
            tuning=[l for l in (root/'server.log').read_text(errors='replace')[boundary:].splitlines() if 'Triton autotuning for function' in l]
            state['short_4k_rounds'].append(dict(round=index+1,duration_seconds=duration,total_tokens_per_second=64*4608/duration,
                output_tokens_per_second=64*512/duration,mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in responses),
                successful_requests=64,responses=responses,tuning_records=tuning));save()
            phase_marker(f'flush_{index+1}')
            state['flush_canaries'].append(request(128,16,2026101699,777));save()
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
            values=[r['total_tokens_per_second'] for r in state['short_4k_rounds']] 
            state.update(phase='complete',mean_total_tok_s=statistics.mean(values),spread_percent=100*(max(values)-min(values))/statistics.mean(values))
        except Exception as error:state.update(phase='audit_failed',error=repr(error))
    save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
