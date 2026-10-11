#!/usr/bin/env python3
"""Short A/B service probe for startup GEMM bucketing; never scored as official."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise RuntimeError('Refusing to overwrite completed experiment')
    env = dict(os.environ, VLLM_PLUGINS='fl', PYTHONUNBUFFERED='1',
               VLLM_FL_USE_FLAGGEMS_ATTN='0', FLAGGEMS_ILUVATAR_COMPACT_VARLEN='0',
               VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN='0', TRITON_PRINT_AUTOTUNING='1',
               VLLM_FL_ILUVATAR_PREFILL_BUCKETS=args.buckets,
               VLLM_FL_ILUVATAR_LOGITS_GATHER=args.gather)
    command = ['vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
               '--compilation-config', '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
               '--served-model-name', 'minicpm', '--gpu-memory-utilization', '0.85',
               '--max-model-len', '131072']
    state = dict(phase='preparing', runner_pid=os.getpid(), buckets=args.buckets, gather=args.gather,
                 started_at_utc=datetime.now(timezone.utc).isoformat(), command=command,
                 rare_cases=[], short_4k_rounds=[], scored=False,
                 workload={'input_tokens': 4096, 'output_tokens': 64, 'concurrency': 16,
                           'requests_per_round': 16, 'rounds': 5},
                 note='Five short 4K rounds, not the official 4K/1K 256-request benchmark.')
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
        for index, length in enumerate([565, 577, 599, 1537, 1561]):
            deadline()
            before = len(log_path.read_text())
            result = request(length, 16, 2026101120 + index, False, 0)
            result['tune_records'] = tune_records(log_path.read_text()[before:])
            state['rare_cases'].append(result)
            save()
        for round_index in range(5):
            deadline()
            state['phase'] = f'short_4k_round_{round_index + 1}'
            save()
            before = len(log_path.read_text())
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(request, 4096, 64, 2026101200 + 100 * round_index + i,
                                       True, 1.0) for i in range(16)]
                responses = [future.result() for future in futures]
            duration = time.perf_counter() - start
            state['short_4k_rounds'].append(dict(round=round_index + 1, duration_seconds=duration,
                    total_tokens_per_second=16 * (4096 + 64) / duration,
                    output_tokens_per_second=16 * 64 / duration,
                    mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in responses),
                    successful_requests=len(responses),
                    tuning_records=tune_records(log_path.read_text()[before:]), responses=responses))
            save()
        state['runtime_tune_records'] = tune_records(log_path.read_text()[boundary:])
        values = [r['total_tokens_per_second'] for r in state['short_4k_rounds'][1:]]
        state['later_four_stability'] = dict(mean=statistics.mean(values),
                    min=min(values), max=max(values), population_stddev=statistics.pstdev(values),
                    spread_percent=100 * (max(values) - min(values)) / statistics.mean(values))
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
