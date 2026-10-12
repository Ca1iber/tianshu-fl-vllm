#!/usr/bin/env python3
"""Process-only native GQA prefill service trial."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
import traceback
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--buckets', choices=['0', '1'], required=True)
    parser.add_argument('--gather', choices=['0', '1'], default='0')
    parser.add_argument('--hybrid', choices=['1'], default='1')
    parser.add_argument('--adaptive', choices=['0','1'], default='0')
    parser.add_argument('--max-batched-tokens', type=int, choices=[2048,4096,8192], required=True)
    parser.add_argument('--tile',choices=['0','1'],required=True)
    parser.add_argument('--native-prefill',choices=['0','1'],required=True)
    parser.add_argument('--single-probe-output', type=int, default=0)
    parser.add_argument('--output-tokens',type=int,default=1024)
    parser.add_argument('--concurrency', type=int, default=64)
    parser.add_argument('--requests', type=int, default=64)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if min(args.output_tokens, args.concurrency, args.requests) < 1 or args.rounds < 2:
        parser.error('Positive workload sizes and at least two rounds are required')
    if args.buckets != '0' or args.gather != '0':
        raise RuntimeError('Padding/gather flags must stay off for a token-budget trial')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise RuntimeError('Refusing to overwrite completed experiment')
    env = dict(os.environ, VLLM_PLUGINS='fl', PYTHONUNBUFFERED='1',
               VLLM_FL_USE_FLAGGEMS_ATTN='0', FLAGGEMS_ILUVATAR_COMPACT_VARLEN=args.hybrid,
               VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN='0', TRITON_PRINT_AUTOTUNING='1',
               VLLM_FL_ILUVATAR_PREFILL_BUCKETS=args.buckets,
               VLLM_FL_ILUVATAR_LOGITS_GATHER=args.gather)
    env.update(VLLM_FL_ILUVATAR_HYBRID_ATTENTION=args.hybrid,
               VLLM_FL_ILUVATAR_HYBRID_STATS_PATH=str(out/'routing_stats.json'))
    env['VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS'] = args.adaptive
    env.pop('TIANSHU_PACK_PROBE_MODE',None)
    env.pop('PYTHONPATH',None)
    env.pop('TIANSHU_FA2_TILE_PROBE',None)
    env.pop('TIANSHU_FA2_TILE_RECORD_DIR',None)
    if args.tile == '1':
        env['PYTHONPATH']=str(Path(__file__).resolve().parent/'hooks')
        env['TIANSHU_FA2_TILE_PROBE']='1'
        env['TIANSHU_FA2_TILE_RECORD_DIR']=str(out/'tile_records')
    original_hook=Path(__file__).resolve().parent.parent/'tianshu_fa2_tiles_20261012/hooks/sitecustomize.py'
    env['PYTHONPATH']=str(original_hook.parent)
    env.pop('TIANSHU_NATIVE_PREFILL_PROBE',None)
    if args.tile!='1' or args.max_batched_tokens!=2048:
        raise RuntimeError('Prefill GEMM keeps tile1/budget2048')
    if args.native_prefill=='1':
        env['PYTHONPATH']=str(Path(__file__).resolve().parent/'hooks')
        env['TIANSHU_NATIVE_PREFILL_PROBE']='1'
        env['TIANSHU_NATIVE_PREFILL_RECORD_DIR']=str(out/'native_prefill_records')
    command = ['vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
               '--compilation-config', '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
               '--served-model-name', 'minicpm', '--gpu-memory-utilization', '0.85',
               '--max-model-len', '131072']
    command += ['--max-num-batched-tokens', str(args.max_batched_tokens)]
    state = dict(native_prefill=args.native_prefill,tile=args.tile, max_batched_tokens=args.max_batched_tokens, adaptive=args.adaptive, single_decode_rounds=[], hybrid=args.hybrid, phase='preparing', runner_pid=os.getpid(), buckets=args.buckets, gather=args.gather,
                 started_at_utc=datetime.now(timezone.utc).isoformat(), command=command,
                 rare_cases=[], short_4k_rounds=[], scored=False,
                 workload={'input_tokens': 4096, 'output_tokens': args.output_tokens,
                           'concurrency': args.concurrency, 'requests_per_round': args.requests,
                           'rounds': args.rounds, 'excluded_rounds': [1]},
                 note='Short 4K experiment excluding round one; not the official 256-request average score.')
    server = None

    class StopRequested(BaseException):
        pass

    def stop_signal(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Close streaming requests before ThreadPoolExecutor waits for workers.
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
        raise StopRequested(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)

    stop_at = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])

    def lease_watchdog():
        while datetime.now(timezone.utc) < stop_at:
            time.sleep(5)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(10)
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGKILL)

    threading.Thread(target=lease_watchdog, daemon=True).start()

    def save():
        state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = out / 'state.json.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(out / 'state.json')
        print(json.dumps({'phase': state['phase'], 'rare': len(state['rare_cases']),
                          'rounds': len(state['short_4k_rounds']), 'error': state.get('error')}), flush=True)

    def deadline():
        path = Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
        stop = datetime.fromisoformat(json.loads(path.read_text())['stop_at_utc'])
        if datetime.now(timezone.utc) >= stop:
            raise StopRequested('Reservation deadline reached')

    def request(length, output, seed, stream, temperature):
        rng = random.Random(seed)
        payload = dict(model='minicpm', prompt=[rng.randrange(512, 120000) for _ in range(length)],
                       max_tokens=output, ignore_eos=True, temperature=temperature, top_p=.95,
                       stream=stream)
        if stream:
            payload['stream_options'] = {'include_usage': True}
        req = urllib.request.Request('http://127.0.0.1:9031/v1/completions',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
        started = time.perf_counter()
        text = ''
        ttft = None
        usage = None
        with urllib.request.urlopen(req, timeout=900) as response:
            if stream:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith('data: ') or line == 'data: [DONE]':
                        continue
                    event = json.loads(line[6:])
                    choices = event.get('choices', [])
                    if choices:
                        chunk = choices[0].get('text', '')
                        text += chunk
                        if chunk and ttft is None:
                            ttft = time.perf_counter() - started
                    if event.get('usage'):
                        usage = event['usage']
            else:
                result = json.load(response)
                usage = result['usage']
                text = result['choices'][0]['text']
        elapsed = time.perf_counter() - started
        if usage is None or usage['prompt_tokens'] != length or usage['completion_tokens'] != output:
            raise RuntimeError(f'Incorrect request usage: {usage}')
        if stream and ttft is None:
            raise RuntimeError('Stream contained no text token')
        return dict(input_tokens=length, output_tokens=output, seed=seed, usage=usage,
                    wall_seconds=elapsed, ttft_seconds=ttft, text=text)

    def tune_records(text):
        return [line for line in text.splitlines() if 'Triton autotuning for function' in line]

    try:
        try:
            connection = socket.create_connection(('127.0.0.1', 9031), timeout=2)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError('Port 9031 is occupied; refusing to replace another service')
        hardware = subprocess.run(['ixsmi'], capture_output=True, text=True, timeout=15)
        (out / 'hardware_before.txt').write_text(hardware.stdout + hardware.stderr)
        if hardware.returncode or 'No running processes found' not in hardware.stdout:
            raise RuntimeError('GPU must be idle before a serialized budget trial')
        installed = Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker')
        source_dir = out / 'source_snapshot'
        source_dir.mkdir(exist_ok=True)
        sources = {}
        for name in ('worker.py', 'iluvatar_hybrid_attention.py', 'iluvatar_adaptive_decode.py'):
            source = installed / name
            contents = source.read_bytes()
            (source_dir / name).write_bytes(contents)
            sources[name] = {'path': str(source), 'sha256': hashlib.sha256(contents).hexdigest()}
        state['installed_sources'] = sources
        if args.tile == '1':
            hook=(Path(__file__).with_name('hooks')/'sitecustomize.py') if args.native_prefill=='1' else original_hook
            contents=hook.read_bytes();(source_dir/'tile_sitecustomize.py').write_bytes(contents)
            state['tile_hook']={'path':str(hook),'sha256':hashlib.sha256(contents).hexdigest()}
        kernel=Path(__file__).resolve().parent.parent/'tianshu_native_prefill_20261012/zero_initialization.py'
        state['zero_initialization']={'path':str(kernel),'sha256':hashlib.sha256(kernel.read_bytes()).hexdigest()}
        (source_dir/'zero_initialization.py').write_bytes(kernel.read_bytes())
        (source_dir/'original_tile_sitecustomize.py').write_bytes(original_hook.read_bytes())
        state['environment_flags'] = {key: value for key, value in env.items()
                                      if key.startswith(('VLLM_FL_', 'FLAGGEMS_ILUVATAR_'))}
        save()
        with (out / 'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd='/workspace', env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        state.update(phase='starting_server', server_pid=server.pid)
        save()
        started = time.monotonic()
        while True:
            deadline()
            if server.poll() is not None:
                raise RuntimeError(f'Server exited: {server.returncode}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models', timeout=3) as response:
                    models = json.load(response)
                if any(model.get('id') == 'minicpm' for model in models.get('data', [])):
                    break
            except Exception:
                pass
            if time.monotonic() - started > 2400:
                raise TimeoutError('Startup exceeded 40 minutes')
            time.sleep(3)
        state['startup_seconds'] = time.monotonic() - started
        state['ready_at_utc'] = datetime.now(timezone.utc).isoformat()
        log_path = out / 'server.log'
        boundary = len(log_path.read_text())
        state['startup_tune_records'] = tune_records(log_path.read_text())
        state['phase'] = 'rare_shapes'
        save()
        for index, length in enumerate([565, 577, 599, 1537, 1561, 2048, 4096]):
            deadline()
            before = len(log_path.read_text())
            result = request(length, 16, 2026101120 + index, False, 0)
            result['tune_records'] = tune_records(log_path.read_text()[before:])
            state['rare_cases'].append(result)
            save()
        for round_index in range(args.rounds):
            deadline()
            state['phase'] = f'short_4k_round_{round_index + 1}'
            save()
            before = len(log_path.read_text())
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(request, 4096, args.output_tokens,
                                       2026101200 + max(100, args.requests) * round_index + i,
                                       True, 1.0) for i in range(args.requests)]
                responses = [future.result() for future in futures]
            duration = time.perf_counter() - start
            state['short_4k_rounds'].append(dict(round=round_index + 1, duration_seconds=duration,
                    total_tokens_per_second=args.requests * (4096 + args.output_tokens) / duration,
                    output_tokens_per_second=args.requests * args.output_tokens / duration,
                    mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in responses),
                    successful_requests=len(responses),
                    tuning_records=tune_records(log_path.read_text()[before:]), responses=responses))
            save()
        state['runtime_tune_records'] = tune_records(log_path.read_text()[boundary:])
        values = [r['total_tokens_per_second'] for r in state['short_4k_rounds'][1:]]
        state['measured_round_stability'] = dict(mean=statistics.mean(values),
                    count=len(values), excluded_rounds=[1],
                    min=min(values), max=max(values), population_stddev=statistics.pstdev(values),
                    spread_percent=100 * (max(values) - min(values)) / statistics.mean(values))
        if args.rounds == 5:
            state['later_four_stability'] = state['measured_round_stability']
        if args.single_probe_output:
            state['single_decode_workload'] = {'input_tokens':4096,'output_tokens':args.single_probe_output,'concurrency':1,'requests_per_round':1,'rounds':3,'excluded_rounds':[1]}
            for index in range(3):
                deadline()
                state['phase'] = f'single_decode_round_{index+1}'
                save()
                response = request(4096, args.single_probe_output, 2026101210+index, True, 1.0)
                state['single_decode_rounds'].append(response)
                save()
        if args.tile == '1':
            records=list((out/'tile_records').glob('tile_*.json'))
            if not records and args.native_prefill=='0':raise RuntimeError('FA2 tile override did not route on actual model')
            state['tile_routing_records']=[json.loads(p.read_text()) for p in records]
        if args.native_prefill=='1':
            files=list((out/'native_prefill_records').glob('native_prefill_*.json'))
            records=[json.loads(f.read_text()) for f in files]
            if not records:
                raise RuntimeError('Native prefill did not route on actual model')
            state['native_prefill_routing_records']=records

        state['phase'] = 'complete'
    except StopRequested as error:
        state.update(phase='stopped', error=str(error))
    except BaseException as error:
        state.update(phase='failed', error=str(error), traceback=traceback.format_exc())
        print(state['traceback'], flush=True)
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=5)
        hardware = subprocess.run(['ixsmi'], capture_output=True, text=True, timeout=15)
        (out / 'hardware_after.txt').write_text(hardware.stdout + hardware.stderr)
        state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        (out / 'result.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    return 0 if state['phase'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
