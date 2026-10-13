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
    parser.add_argument('--input-tokens',type=int,default=4096)
    parser.add_argument('--watermark',type=float,default=0.0)
    parser.add_argument('--hist-count-bits',choices=['off','64','32','16','8'],default='off')
    parser.add_argument('--decode-graph-max',type=int,choices=[0,64],default=0)
    parser.add_argument('--sort-buffer-mode',choices=['original','index32','scan32','both32'],default='original')
    parser.add_argument('--address32',choices=['0','1'],default='0')
    parser.add_argument('--request-timeout',type=int,default=900)
    parser.add_argument('--concurrency', type=int, default=64)
    parser.add_argument('--requests', type=int, default=64)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--decode-launch',choices=['0','1'],default='0')
    parser.add_argument('--stream-interval',type=int,choices=[1,4],default=1)
    parser.add_argument('--bucket-trace',choices=['0','1','2'],default='0')
    args = parser.parse_args()
    if min(args.input_tokens, args.output_tokens, args.concurrency, args.requests, args.request_timeout) < 1 or args.rounds < 2:
        parser.error('Positive workload sizes and at least two rounds are required')
    if not 0 <= args.watermark < 1 or args.input_tokens + args.output_tokens > 131072:
        parser.error('Watermark must be in [0,1), and the workload must fit the model length')
    if args.gather == '1' and args.buckets != '1':
        raise RuntimeError('Fixed gather requires the independently checked bucket profile')
    if args.buckets == '1' and (args.max_batched_tokens != 2048 or args.native_prefill != '1'):
        raise RuntimeError('Bucket confirmation requires the verified native profile at budget2048')
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
    if args.tile!='1':
        raise RuntimeError('Native Prefill budget confirmation requires the verified tile')
    if args.native_prefill=='1':
        env['PYTHONPATH']=str(Path(__file__).resolve().parent/'hooks')
        env['TIANSHU_NATIVE_PREFILL_PROBE']='1'
        env['TIANSHU_NATIVE_PREFILL_RECORD_DIR']=str(out/'native_prefill_records')
    if args.decode_launch=='1':
        assert args.native_prefill=='1' and args.tile=='1'
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'tianshu_decode_launch_20261012/hooks')
        env['TIANSHU_DECODE_LAUNCH_PROBE']='1'
        env['TIANSHU_DECODE_LAUNCH_RECORD_DIR']=str(out/'decode_launch_records')
    if args.bucket_trace!='0':
        assert args.buckets==args.native_prefill==args.decode_launch=='1'
        folder='joint_hooks' if args.bucket_trace=='2' else 'hooks'
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'tianshu_buckets75_recheck_20261012'/folder)
        env['TIANSHU_BUCKET_RECHECK_TRACE']='1'
        env['TIANSHU_BUCKET_TRACE_DIR']=str(out/'bucket_trace_records')
    if args.hist_count_bits!='off':
        assert args.bucket_trace=='2' and args.gather=='1'
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'tianshu_sort_counts_20261013/hooks')
        env['TIANSHU_SORT_COUNT_BITS']=args.hist_count_bits
        env['TIANSHU_SORT_COUNT_RECORD_DIR']=str(out/'sort_count_records')
    else:
        env.pop('TIANSHU_SORT_COUNT_BITS',None)
        env.pop('TIANSHU_SORT_COUNT_RECORD_DIR',None)
    if args.decode_graph_max:
        assert args.hist_count_bits=='off' and args.bucket_trace=='2'
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'tianshu_decode_graphs_20261013/hooks')
        env['TIANSHU_DECODE_GRAPH_MAX_ROWS']=str(args.decode_graph_max)
        env['TIANSHU_DECODE_GRAPH_RECORD_DIR']=str(out/'decode_graph_records')
    else:
        env.pop('TIANSHU_DECODE_GRAPH_MAX_ROWS',None)
        env.pop('TIANSHU_DECODE_GRAPH_RECORD_DIR',None)
    if args.sort_buffer_mode!='original':
        assert args.hist_count_bits=='off' and args.decode_graph_max==0 and args.bucket_trace=='2'
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'tianshu_sort_buffers_20261013/hooks')
        env['TIANSHU_SORT_BUFFER_MODE']=args.sort_buffer_mode
        env['TIANSHU_SORT_BUFFER_RECORD_DIR']=str(out/'sort_buffer_records')
    else:
        env.pop('TIANSHU_SORT_BUFFER_MODE',None)
        env.pop('TIANSHU_SORT_BUFFER_RECORD_DIR',None)
    if args.address32=='1':
        assert args.hist_count_bits=='off' and args.decode_graph_max==0 and args.sort_buffer_mode=='original' and args.bucket_trace=='2'
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'tianshu_attention_offsets_20261013/hooks')
        env['TIANSHU_ADDRESS32_PROBE']='1'
        env['TIANSHU_ADDRESS32_RECORD_DIR']=str(out/'address32_records')
    else:
        env.pop('TIANSHU_ADDRESS32_PROBE',None)
        env.pop('TIANSHU_ADDRESS32_RECORD_DIR',None)
    command = ['vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
               '--compilation-config', '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
               '--served-model-name', 'minicpm', '--gpu-memory-utilization', '0.85',
               '--max-model-len', '131072']
    command += ['--max-num-batched-tokens', str(args.max_batched_tokens)]
    if args.watermark:
        command += ['--watermark',str(args.watermark)]
    if args.stream_interval!=1:
        command += ['--stream-interval',str(args.stream_interval)]
    state = dict(native_prefill=args.native_prefill,tile=args.tile,decode_launch=args.decode_launch,max_batched_tokens=args.max_batched_tokens, adaptive=args.adaptive, single_decode_rounds=[], hybrid=args.hybrid, phase='preparing', runner_pid=os.getpid(), buckets=args.buckets, gather=args.gather,
                 started_at_utc=datetime.now(timezone.utc).isoformat(), command=command,
                 rare_cases=[], short_4k_rounds=[], scored=False,
                 workload={'input_tokens': args.input_tokens, 'output_tokens': args.output_tokens,
                           'concurrency': args.concurrency, 'requests_per_round': args.requests,
                           'rounds': args.rounds, 'excluded_rounds': [1]},
                 note='Local service experiment excluding round one; not an official benchmark score.',
                 watermark=args.watermark, histogram_count_bits=args.hist_count_bits,
                 decode_graph_max=args.decode_graph_max,
                 sort_buffer_mode=args.sort_buffer_mode,
                 address32=args.address32,
                 official_benchmark=False)
    state['stream_interval']=args.stream_interval
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
        with urllib.request.urlopen(req, timeout=args.request_timeout) as response:
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

    def metrics_snapshot(name):
        with urllib.request.urlopen('http://127.0.0.1:9031/metrics', timeout=5) as response:
            contents=response.read().decode()
        (out/f'{name}_metrics.txt').write_text(contents)
        return {'file':f'{name}_metrics.txt',
                'utc':datetime.now(timezone.utc).isoformat()}

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
        for name in ('worker.py', 'iluvatar_hybrid_attention.py', 'iluvatar_adaptive_decode.py',
                     'model_runner.py', 'iluvatar_prefill_buckets.py', 'iluvatar_logits_gather.py'):
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
                                      if key.startswith(('VLLM_FL_', 'FLAGGEMS_ILUVATAR_','TIANSHU_DECODE_'))}
        if args.decode_launch=='1':
            hook=Path(__file__).resolve().parents[1]/'tianshu_decode_launch_20261012/hooks/sitecustomize.py';contents=hook.read_bytes()
            (source_dir/'decode_sitecustomize.py').write_bytes(contents)
            state['decode_hook']={'path':str(hook),'sha256':hashlib.sha256(contents).hexdigest()}
        if args.bucket_trace!='0':
            folder='joint_hooks' if args.bucket_trace=='2' else 'hooks'
            hook=Path(__file__).resolve().parents[1]/'tianshu_buckets75_recheck_20261012'/folder/'sitecustomize.py'
            contents=hook.read_bytes()
            (source_dir/'bucket_trace_sitecustomize.py').write_bytes(contents)
            state['bucket_trace_hook']={'path':str(hook),'sha256':hashlib.sha256(contents).hexdigest()}
            component=Path(__file__).resolve().parents[1]/'tianshu_buckets75_recheck_20261012/hooks/sitecustomize.py'
            (source_dir/'bucket_component_sitecustomize.py').write_bytes(component.read_bytes())
        if args.hist_count_bits!='off':
            count=Path(__file__).resolve().parents[1]/'tianshu_sort_counts_20261013'
            paths={'count_sitecustomize.py':count/'hooks/sitecustomize.py',
                   'histogram_width.py':count/'histogram_width.py',
                   'original_sort.py':Path('/usr/local/lib/python3.12/site-packages/flag_gems/ops/sort.py')}
            state['count_sources']={}
            for name,path in paths.items():
                contents=path.read_bytes();(source_dir/name).write_bytes(contents)
                state['count_sources'][name]={'path':str(path),'sha256':hashlib.sha256(contents).hexdigest()}
        if args.decode_graph_max:
            hook=Path(env['PYTHONPATH'])/'sitecustomize.py';contents=hook.read_bytes()
            (source_dir/'decode_graph_sitecustomize.py').write_bytes(contents)
            state['decode_graph_hook']={'path':str(hook),'sha256':hashlib.sha256(contents).hexdigest()}
        if args.sort_buffer_mode!='original':
            paths={'sort_buffer_sitecustomize.py':Path(env['PYTHONPATH'])/'sitecustomize.py',
                   'buffer_variants.py':Path('/workspace/FlagGems/experiments/tianshu_sort_buffers_20261013/buffer_variants.py'),
                   'original_sort.py':Path('/usr/local/lib/python3.12/site-packages/flag_gems/ops/sort.py')}
            state['sort_buffer_sources']={}
            for name,path in paths.items():
                contents=path.read_bytes();(source_dir/name).write_bytes(contents)
                state['sort_buffer_sources'][name]={'path':str(path),'sha256':hashlib.sha256(contents).hexdigest()}
        if args.address32=='1':
            paths={'address32_sitecustomize.py':Path(env['PYTHONPATH'])/'sitecustomize.py',
                   'address32.py':Path(__file__).resolve().parents[1]/'tianshu_attention_offsets_20261013/address32.py'}
            state['address32_sources']={}
            for name,path in paths.items():
                contents=path.read_bytes();(source_dir/name).write_bytes(contents)
                state['address32_sources'][name]={'path':str(path),'sha256':hashlib.sha256(contents).hexdigest()}
        (source_dir/'run_service_trial.py').write_bytes(Path(__file__).read_bytes())
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
            metric_before=metrics_snapshot(f'round{round_index+1}_before')
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(request, args.input_tokens, args.output_tokens,
                                       2026101200 + max(100, args.requests) * round_index + i,
                                       True, 1.0) for i in range(args.requests)]
                responses = [future.result() for future in futures]
            duration = time.perf_counter() - start
            metric_after=metrics_snapshot(f'round{round_index+1}_after')
            state['short_4k_rounds'].append(dict(round=round_index + 1, duration_seconds=duration,
                    total_tokens_per_second=args.requests * (args.input_tokens + args.output_tokens) / duration,
                    output_tokens_per_second=args.requests * args.output_tokens / duration,
                    mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in responses),
                    successful_requests=len(responses),
                    tuning_records=tune_records(log_path.read_text()[before:]), responses=responses,
                    metrics_before=metric_before, metrics_after=metric_after))
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
        if args.hist_count_bits!='off':
            records=[json.loads(f.read_text()) for f in (out/'sort_count_records').glob('route_*.json')]
            assert records, 'Count override did not route on the actual model'
            state['sort_count_routing_records']=records
        if args.decode_graph_max:
            records=[json.loads(f.read_text()) for f in (out/'decode_graph_records').glob('configured_*.json')]
            assert records and all(max(r['decode_graph_sizes'])==64 and
                                   len(r['prefill_warmup_sizes'])==75 for r in records)
            warmups=[json.loads(f.read_text()) for f in (out/'bucket_trace_records').glob('warmup_completed_*.json')]
            assert any(r['complete'] and len(r['calls'])==75 for r in warmups)
            state['decode_graph_configuration_records']=records
        if args.sort_buffer_mode!='original':
            records=[json.loads(f.read_text()) for f in (out/'sort_buffer_records').glob('route_*.json')]
            assert records and all(r['public_index_dtype']=='torch.int64' for r in records)
            state['sort_buffer_routing_records']=records
        if args.address32=='1':
            records=[json.loads(f.read_text()) for f in (out/'address32_records').glob('route_*.json')]
            assert records and all(4<=r['batch']<=48 and max(r['bounds'])<2**31 for r in records)
            state['address32_routing_records']=records

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
