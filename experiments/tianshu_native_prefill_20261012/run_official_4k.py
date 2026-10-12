# SPDX-License-Identifier: Apache-2.0
"""Run the unmodified official4K after verified native-prefill service gain and Math105."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request


BENCH = Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
BENCH_SHA = '0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
SOURCE = Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker/iluvatar_hybrid_attention.py')
LEASE = Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-trial', type=Path, required=True)
    parser.add_argument('--candidate-trial', type=Path, required=True)
    parser.add_argument('--max-batched-tokens', type=int, choices=[2048,4096,8192], required=True)
    parser.add_argument('--math-results', type=Path, required=True)
    parser.add_argument('--minimum-gain-percent', type=float, default=2)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'state.json').exists():
        raise RuntimeError('Refusing to overwrite an existing official trial')
    trials = [json.loads((directory / 'result.json').read_text())
              for directory in (args.control_trial, args.candidate_trial)]
    assert all(t['phase'] == 'complete' for t in trials)
    assert trials[0]['workload'] == trials[1]['workload']
    assert trials[1]['adaptive'] == '1' and trials[1]['tile'] == '1'
    assert trials[1].get('max_batched_tokens',2048) == args.max_batched_tokens
    values = [[r['total_tokens_per_second'] for r in t['short_4k_rounds'][1:]] for t in trials]
    gain = 100 * (statistics.mean(values[1]) / statistics.mean(values[0]) - 1)
    if gain < args.minimum_gain_percent or min(values[1]) <= max(values[0]):
        raise RuntimeError(f'Short improvement is not clear enough: {gain:.3f}%')
    audit = json.loads((args.math_results / 'accuracy_audit.json').read_text())
    assert audit['complete'] and audit['total'] == 105 and not audit['reused_original_accuracy']
    assert audit['accuracy'] > .95 and audit['correct'] / audit['total'] > .95
    assert audit['adaptive_flag'] == '1'
    assert audit.get('max_batched_tokens',2048) == args.max_batched_tokens
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == audit['hybrid_sha256']
    adaptive=SOURCE.with_name('iluvatar_adaptive_decode.py')
    assert hashlib.sha256(adaptive.read_bytes()).hexdigest() == audit['adaptive_sha256']
    tile_configuration={'BLOCK_M':128,'BLOCK_N':128,'num_warps':8,'num_stages':2}
    assert audit['tile_configuration']==tile_configuration
    assert audit['experimental_candidate']=='native-prefill' and audit['candidate_routing_records']
    hook=Path(__file__).resolve().parents[1]/'tianshu_native_prefill_20261012/hooks/sitecustomize.py'
    assert hashlib.sha256(hook.read_bytes()).hexdigest()==audit['candidate_hook']['sha256']
    assert hook.read_bytes()==(args.math_results/'source_snapshot/candidate_sitecustomize.py').read_bytes()
    assert trials[1]['tile_hook']['sha256']==audit['candidate_hook']['sha256']
    snapshot=args.math_results/'source_snapshot/worker.py'
    assert snapshot.read_bytes() == SOURCE.with_name('worker.py').read_bytes()
    for name, data in trials[1]['installed_sources'].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest() == data['sha256']
    assert hashlib.sha256(BENCH.read_bytes()).hexdigest() == BENCH_SHA
    assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
    state = {'phase': 'preparing', 'runner_pid': os.getpid(), 'official_benchmark': True,
             'test_cases': [[4096, 1024, 64, 256]], 'runs': 4, 'skip_first': 1,
             'started_utc': datetime.now(timezone.utc).isoformat(),
             'short_gain_percent': gain, 'official_script_sha256': BENCH_SHA,
             'max_batched_tokens':args.max_batched_tokens,
             'tile_configuration':tile_configuration, 'tile_hook':audit['tile_hook'], 'candidate_hook':audit['candidate_hook'],
             'experimental_kernel':audit['experimental_kernel'], 'experimental_candidate':'native-prefill',
             'adaptive_sha256':audit['adaptive_sha256'],
             'worker_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
             'hybrid_source': str(SOURCE), 'hybrid_sha256': audit['hybrid_sha256'],
             'independent_math': {'correct': audit['correct'], 'total': audit['total'],
                                  'accuracy': audit['accuracy'],
                                  'official_requirement': 'accuracy > 0.95',
                                  'official_requirement_passed': True},
             'candidate_quality_passed': True}
    server = benchmark = None

    class StopRequested(BaseException):
        pass

    def terminate(process):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
        if benchmark is not None and benchmark.poll() is None:
            os.killpg(benchmark.pid, signal.SIGTERM)
        raise StopRequested(str(signum))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stop_at = datetime.fromisoformat(json.loads(LEASE.read_text())['stop_at_utc'])
    state['stop_at_utc'] = stop_at.isoformat()

    def watchdog():
        while datetime.now(timezone.utc) < stop_at:
            time.sleep(5)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=watchdog, daemon=True).start()

    def save():
        temp = root / 'state.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(root / 'state.json')
        print(json.dumps({'phase': state['phase'], 'completed_runs': state.get('completed_runs', 0),
                          'error': state.get('error')}), flush=True)

    def rows():
        files = list((root / 'benchmark_results').glob('raw_runs_*.csv'))
        if not files:
            return []
        assert len(files) == 1
        with files[0].open(newline='') as f:
            return list(csv.DictReader(f))

    try:
        if datetime.now(timezone.utc) >= stop_at:
            raise StopRequested('Reservation ended')
        try:
            connection = socket.create_connection(('127.0.0.1', 9031), timeout=2)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError('Port9031 occupied; refusing to replace another service')
        env = dict(os.environ, VLLM_PLUGINS='fl', PYTHONUNBUFFERED='1',
                   VLLM_FL_ILUVATAR_HYBRID_ATTENTION='1', FLAGGEMS_ILUVATAR_COMPACT_VARLEN='1',
                   VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS='1',
                   VLLM_FL_USE_FLAGGEMS_ATTN='0', VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN='0',
                   VLLM_FL_ILUVATAR_PREFILL_BUCKETS='0', VLLM_FL_ILUVATAR_LOGITS_GATHER='0')
        for name in ('PYTHONPATH', 'TIANSHU_PACK_PROBE_MODE', 'VLLM_FL_ILUVATAR_HYBRID_STATS_PATH'):
            env.pop(name, None)
        env['PYTHONPATH']=str(hook.parent)
        env['TIANSHU_FA2_TILE_PROBE']='1'
        env['TIANSHU_FA2_TILE_RECORD_DIR']=str(root/'tile_records')
        env['TIANSHU_NATIVE_PREFILL_PROBE']='1'
        env['TIANSHU_NATIVE_PREFILL_RECORD_DIR']=str(root/'native_prefill_records')
        state['flags'] = {k: v for k, v in env.items() if k.startswith(('VLLM_FL_', 'FLAGGEMS_ILUVATAR_', 'VLLM_PLUGINS'))}
        state['packages'] = {name: importlib.metadata.version(name)
                             for name in ('torch', 'vllm', 'vllm-plugin-fl', 'flag-gems', 'evalscope')}
        (root / 'hardware_before.txt').write_text(subprocess.check_output(['ixsmi'], text=True))
        command = ['vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
                   '--compilation-config', '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
                   '--served-model-name', 'minicpm', '--gpu-memory-utilization', '0.85',
                   '--max-model-len', '131072']
        command += ['--max-num-batched-tokens', str(args.max_batched_tokens)]
        state['server_command'] = command
        with (root / 'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd='/workspace', env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        state.update(phase='starting', server_pid=server.pid)
        save()
        limit = time.monotonic() + 2400
        while True:
            if server.poll() is not None:
                raise RuntimeError('Server exited during startup')
            if time.monotonic() > limit:
                raise TimeoutError('Startup exceeded40min')
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models', timeout=3) as response:
                    models = json.load(response)
                if any(m['id'] == 'minicpm' for m in models['data']):
                    break
            except Exception:
                pass
            time.sleep(3)
        command = [sys.executable, '-u', str(BENCH), '--model', '/workspace/MiniCPM5-2B',
                   '--served-model-name', 'minicpm', '--port', '9031',
                   '--test-cases', '[[4096,1024,64,256]]']
        state['benchmark_command'] = command
        state['benchmark_working_directory'] = str(root)
        with (root / 'benchmark.log').open('w') as log:
            benchmark = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        state.update(phase='benchmarking', benchmark_pid=benchmark.pid)
        save()
        while benchmark.poll() is None:
            completed = rows()
            if state.get('completed_runs') != len(completed):
                state.update(completed_runs=len(completed), raw_runs=completed)
                save()
            time.sleep(5)
        state['benchmark_exit_code'] = benchmark.returncode
        assert benchmark.returncode == 0
        completed = rows()
        assert len(completed) == 4
        for row in completed:
            assert row['Run Status'] == 'SUCCESS' and float(row['Successful Requests']) == 256
            assert [int(row[k]) for k in ('Prefill', 'Decode', 'Conc', 'Num Prompts')] == [4096, 1024, 64, 256]
            assert float(row['Total Input Tokens']) == 1048576 and float(row['Total Output Tokens']) == 262144
        summaries = list((root / 'benchmark_results').glob('summary_*.csv'))
        assert len(summaries) == 1
        with summaries[0].open(newline='') as f:
            summary = list(csv.DictReader(f))
        assert len(summary) == 1
        for key in ('Total tok/s', 'Mean TTFT (ms)', 'Output tok/s'):
            assert abs(float(summary[0][key]) - statistics.mean(float(r[key]) for r in completed[1:])) < .011
        assert hashlib.sha256(BENCH.read_bytes()).hexdigest() == BENCH_SHA
        records=list((root/'tile_records').glob('tile_*.json'))
        native_records=[json.loads(p.read_text()) for p in (root/'native_prefill_records').glob('native_prefill_*.json')]
        assert native_records, 'Actual model did not use quality-verified native prefill'
        assert all(x['kernel_sha256']==audit['experimental_kernel']['sha256'] for x in native_records)
        expected_generated={x['generated_kernel_sha256'] for x in audit['candidate_routing_records']}
        assert all(x['generated_kernel_sha256'] in expected_generated for x in native_records)
        state['candidate_routing_records']=native_records
        state['tile_routing_records']=[json.loads(p.read_text()) for p in records]
        state.update(phase='complete', completed_runs=4, raw_runs=completed, official_summary=summary[0])
    except StopRequested as error:
        state.update(phase='stopped', error=str(error))
    except BaseException as error:
        state.update(phase='failed', error=str(error), traceback=traceback.format_exc())
    finally:
        terminate(benchmark)
        terminate(server)
        hardware = subprocess.check_output(['ixsmi'], text=True)
        (root / 'hardware_after.txt').write_text(hardware)
        state['gpu_idle_after'] = 'No running processes found' in hardware
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        (root / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
    return 0 if state['phase'] == 'complete' and state['gpu_idle_after'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
