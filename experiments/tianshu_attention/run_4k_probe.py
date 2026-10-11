#!/usr/bin/env python3
"""One 4K official-shaped serving trial; no 16K or four-round scoring loop.

Starts and stops only its own service. Core model/server/generation settings
match the user-provided competition command. Candidate selects an existing
attention implementation through VLLM_FL_USE_FLAGGEMS_ATTN.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attention-backend', choices=['triton', 'flaggems'], required=True)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--compact-grid', action='store_true')
    parser.add_argument('--split-mixed', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise RuntimeError('Refusing to overwrite an existing experiment result')
    bench = Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
    deadline_file = Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    env = dict(os.environ, VLLM_PLUGINS='fl', PYTHONUNBUFFERED='1',
               VLLM_FL_USE_FLAGGEMS_ATTN='1' if args.attention_backend == 'flaggems' else '0',
               FLAGGEMS_ILUVATAR_COMPACT_VARLEN='1' if args.compact_grid else '0',
               VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN='1' if args.split_mixed else '0')
    server_command = ['vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
                      '--compilation-config', '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
                      '--served-model-name', 'minicpm', '--gpu-memory-utilization', '0.85',
                      '--max-model-len', '131072']
    command = ['vllm', 'bench', 'serve', '--backend', 'vllm', '--model', 'minicpm',
               '--tokenizer', '/workspace/MiniCPM5-2B', '--endpoint', '/v1/completions',
               '--host', 'localhost', '--port', '9031', '--dataset-name', 'random',
               '--ignore-eos', '--random-input-len', '4096', '--random-output-len', '1024',
               '--max-concurrency', '64', '--num-prompts', '256']
    state = dict(phase='preparing', runner_pid=os.getpid(), attention_backend=args.attention_backend,
                 started_at_utc=datetime.now(timezone.utc).isoformat(), server_command=server_command,
                 benchmark_command=command, workload=[4096, 1024, 64, 256], scored=False,
                 note='Single official-shaped trial; not the official four-run mean.',
                 environment_overrides={name: env[name] for name in
                                        ['VLLM_PLUGINS', 'VLLM_FL_USE_FLAGGEMS_ATTN', 'FLAGGEMS_ILUVATAR_COMPACT_VARLEN',
                                         'VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN']},
                 official_script_sha256=hashlib.sha256(bench.read_bytes()).hexdigest())
    server = client = None

    class StopRequested(BaseException):
        pass

    def stop_signal(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise StopRequested(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)

    def save():
        state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = out / 'state.json.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(out / 'state.json')
        print(json.dumps({'phase': state['phase'], 'error': state.get('error')}), flush=True)

    def deadline():
        if deadline_file.exists():
            stop = datetime.fromisoformat(json.loads(deadline_file.read_text())['stop_at_utc'])
            if datetime.now(timezone.utc) >= stop:
                raise StopRequested('Reservation deadline reached')

    def cleanup(process):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    try:
        if args.attention_backend == 'flaggems':
            if args.validation is None or not json.loads(args.validation.read_text())['passed']:
                raise RuntimeError('Candidate requires a successful independent operator validation')
            validation = json.loads(args.validation.read_text())
            if args.compact_grid and not all(case.get('compact_bitwise_equal_to_direct') for case in validation['cases']):
                raise RuntimeError('Compact candidate is missing bitwise validation')
            if args.split_mixed and (
                not any(case.get('split_active') for case in validation['cases'])
                or not all(case.get('metadata_and_forward_adapter_passed') for case in validation['cases'])
            ):
                raise RuntimeError('Split candidate is missing metadata/forward validation')
            state['operator_validation'] = str(args.validation.resolve())
            state['operator_validation_sha256'] = hashlib.sha256(args.validation.read_bytes()).hexdigest()
        try:
            connection = socket.create_connection(('127.0.0.1', 9031), timeout=2)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError('Port 9031 is occupied; refusing to replace another service')
        for name in ['vllm_fl', 'flag_gems', 'vllm']:
            spec = importlib.util.find_spec(name)
            state.setdefault('module_paths', {})[name] = spec.origin if spec else None
        state['installed_sources'] = {}
        for package, names in {
            'flag_gems': ['ops/flash_api.py', 'ops/flash_kernel.py'],
            'vllm_fl': ['dispatch/backends/flaggems/impl/attention.py',
                        'dispatch/backends/flaggems/impl/iluvatar_mixed_attention.py'],
        }.items():
            root = Path(state['module_paths'][package]).parent
            for name in names:
                path = root / name
                if path.exists():
                    state['installed_sources'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        save()
        with (out / 'server.log').open('w') as log:
            server = subprocess.Popen(server_command, cwd='/workspace', env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        state.update(phase='starting_server', server_pid=server.pid)
        save()
        limit = time.monotonic() + 1200
        while True:
            deadline()
            if server.poll() is not None:
                raise RuntimeError(f'Server exited with {server.returncode}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models', timeout=3) as response:
                    models = json.load(response)
                if any(model.get('id') == 'minicpm' for model in models.get('data', [])):
                    break
            except Exception:
                pass
            if time.monotonic() >= limit:
                raise TimeoutError('Server did not start within 20 minutes')
            time.sleep(3)
        state['ready_at_utc'] = datetime.now(timezone.utc).isoformat()
        # Warm a unique prompt, without giving the measured dataset a cached prefix.
        state['phase'] = 'warmup'
        save()
        generator = random.Random(2026101119)
        payload = dict(model='minicpm', prompt=[generator.randrange(512, 120000) for _ in range(4096)],
                       max_tokens=16, ignore_eos=True, temperature=0, stream=False)
        request = urllib.request.Request('http://127.0.0.1:9031/v1/completions', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=300) as response:
            warmup = json.load(response)
        if warmup['usage']['prompt_tokens'] != 4096 or warmup['usage']['completion_tokens'] != 16:
            raise RuntimeError(f'Incorrect warmup usage: {warmup.get("usage")}')
        state['warmup_usage'] = warmup['usage']
        state['phase'] = 'benchmark_running'
        save()
        with (out / 'benchmark.log').open('w') as log:
            client = subprocess.Popen(command, cwd='/workspace', env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        state['benchmark_pid'] = client.pid
        save()
        while client.poll() is None:
            deadline()
            if server.poll() is not None:
                raise RuntimeError('Server exited during benchmark')
            time.sleep(5)
        if client.returncode:
            raise RuntimeError(f'Benchmark exited with {client.returncode}')
        spec = importlib.util.spec_from_file_location('official_benchmark_parser', bench)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = module.extract_metrics((out / 'benchmark.log').read_text())
        state['metrics'] = metrics
        expected = dict(successful_requests=256, total_input_tokens=256 * 4096,
                        total_output_tokens=256 * 1024)
        for name, value in expected.items():
            if metrics.get(name) != value:
                raise RuntimeError(f'{name}: expected {value}, got {metrics.get(name)}')
        if not metrics.get('total_token_throughput') or metrics.get('mean_ttft_ms') is None:
            raise RuntimeError('Benchmark metrics are incomplete')
        if hashlib.sha256(bench.read_bytes()).hexdigest() != state['official_script_sha256']:
            raise RuntimeError('Official benchmark source changed during the trial')
        state.update(phase='complete', request_and_token_counts_passed=True)
    except StopRequested as error:
        state.update(phase='paused', error=str(error))
    except BaseException as error:
        state.update(phase='failed', error=str(error), traceback=traceback.format_exc())
        print(state['traceback'], flush=True)
    finally:
        cleanup(client)
        cleanup(server)
        hardware = subprocess.run(['ixsmi'], capture_output=True, text=True, timeout=15)
        (out / 'hardware_after.txt').write_text(hardware.stdout + hardware.stderr)
        state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        (out / 'result.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    return 0 if state['phase'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
