# SPDX-License-Identifier: Apache-2.0
"""Native compiled GPU ngram vs integer candidate: independent component probe."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import statistics
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-at-utc', required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True)
    data = dict(phase='preparing', complete=False, runner_pid=os.getpid(), cases=[],
                events=[], source_snapshots={}, GPU_component_only=True, adopted=False)

    def save():
        temp = root/'result.tmp'
        temp.write_text(json.dumps(data, indent=2)+'\n'); temp.replace(root/'result.json')
        print(json.dumps(dict(phase=data['phase'], cases=len(data['cases']), error=data.get('error'))), flush=True)

    def timeout(signum, frame):
        raise TimeoutError('Component probe reached its deadline')

    def snapshot(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        data['source_snapshots'][str(path)] = hashlib.sha256(raw).hexdigest()
        target = root/'source_snapshot'/path.relative_to('/')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)

    save()
    signal.signal(signal.SIGALRM, timeout)
    remaining = int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds())
    assert remaining > 900
    signal.alarm(min(1800, remaining))
    try:
        hardware = subprocess.check_output(['ixsmi'], text=True)
        assert 'No running processes found' in hardware
        (root/'hardware_before.txt').write_text(hardware)
        base = Path(__file__).resolve().parent
        gate = json.loads((base/'CPU_GATE_32g.json').read_text())
        assert gate['complete'] and gate['exact_cases'] == 32839 and gate['GPU_test'] is False
        assert hashlib.sha256((base/'search.py').read_bytes()).hexdigest() == gate['kernel_source_sha256']
        sys.path.insert(0, str(base.parent/'tianshu_gemm_topp_geo_20261016'))
        from validation_profile_new import prepare
        _, environment, profile, sources = prepare(root, Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
        os.environ.clear(); os.environ.update(environment)
        runpy.run_path(str(Path(profile['PYTHONPATH'])/'sitecustomize.py'))
        import torch
        from vllm.config import CompilationConfig
        from vllm.engine.arg_utils import EngineArgs
        from vllm.forward_context import set_forward_context
        import vllm.v1.spec_decode.ngram_proposer_gpu as native
        import search
        from check_cpu import reference
        torch.set_num_threads(8)
        for name in sources: snapshot(name)
        for path in (Path(__file__), base/'search.py', base/'check_cpu.py', base/'CPU_GATE_32g.json',
                     Path(native.__file__), Path(__import__('vllm.engine.arg_utils', fromlist=['EngineArgs']).__file__)):
            snapshot(path)
        engine = EngineArgs(model='/workspace/MiniCPM5-2B', served_model_name='minicpm',
            max_model_len=131072, gpu_memory_utilization=.85,
            compilation_config=CompilationConfig(cudagraph_mode='FULL_DECODE_ONLY'),
            speculative_config=dict(method='ngram_gpu', num_speculative_tokens=1,
                                    prompt_lookup_min=3, prompt_lookup_max=5))
        configuration = engine.create_engine_config()
        data['phase'] = 'constructing_native_compiled_proposer'; save()
        proposer = native.NgramProposerGPU(configuration, torch.device('cuda'))
        eager_owner = SimpleNamespace(k=1, min_n=3, max_n=5)
        eager_owner._find_first_and_extract_all_n_parallel = MethodType(
            native.NgramGPUKernel._find_first_and_extract_all_n_parallel, eager_owner)
        plans = [(1, 7, 'periodic', False), (8, 32, 'random', False),
                 (63, 4096, 'random', False), (64, 5120, 'periodic', False),
                 (4, 131072, 'random', True), (64, 131071, 'random', False)]
        data['phase'] = 'numerical'; save()
        with torch.inference_mode():
            for case, (n, context, pattern, strided) in enumerate(plans):
                cpu = torch.randint(0, 32768, (n, 131072),
                    generator=torch.Generator().manual_seed(202610191300+case), dtype=torch.int32)
                lengths_cpu = [max(0, context-i%7) for i in range(n)]
                if case == 1: lengths_cpu[:6] = [0, 1, 2, 3, 4, 5]
                if strided:
                    storage = torch.zeros((2*n, 131072), device='cuda', dtype=torch.int64)
                    ids = storage[::2]; ids.copy_(cpu)
                else: ids = cpu.cuda()
                lengths = torch.tensor(lengths_cpu, device='cuda', dtype=torch.int32)
                mask_cpu = [(i % 5 != 4) for i in range(n)]
                mask = torch.tensor(mask_cpu, device='cuda', dtype=torch.bool)
                graph = None
                previous_input = None
                for phase in range(2):
                    for i, length in enumerate(lengths_cpu):
                        if pattern == 'periodic': cpu[i, :length] = torch.arange(length, dtype=torch.int32)%7
                        if length > 12: cpu[i, length-5:length] = cpu[i, 3+phase:8+phase]
                        if phase and length: cpu[i, 0] = (cpu[i, 0]+37) % 32768
                    ids.copy_(cpu)
                    before = ids.clone()
                    if previous_input is not None:
                        assert not torch.equal(before, previous_input), 'Graph inputs did not change'
                    expected = [reference(cpu[i, :length].tolist(), mask_cpu[i])
                                for i, length in enumerate(lengths_cpu)]
                    rng = torch.cuda.get_rng_state().clone()
                    if strided:
                        native_out = native.NgramGPUKernel.forward(eager_owner, lengths, ids, mask)
                    else:
                        with set_forward_context(None, proposer.vllm_config):
                            native_out = proposer.kernel(lengths, ids, mask)
                    candidate = search.forward(lengths, ids, mask)
                    if graph is None:
                        for _ in range(2): search.forward(lengths, ids, mask)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = search.forward(lengths, ids, mask)
                    graph.replay(); torch.cuda.synchronize()
                    for name, result in (('native', native_out), ('candidate', candidate), ('graph', captured)):
                        assert result[0].shape == (n, 1) and result[1].shape == (n,)
                        assert result[0].flatten().cpu().tolist() == [x[0] for x in expected], name
                        assert result[1].cpu().tolist() == [x[1] for x in expected], name
                        assert result[0].dtype == ids.dtype and result[1].dtype == torch.int64
                    assert torch.equal(ids, before)
                    assert torch.equal(rng, torch.cuda.get_rng_state())
                    previous_input = before
                    data['cases'].append(dict(case=case, phase=phase, n=n, context=context,
                        lengths=lengths_cpu, enabled=mask_cpu, expected=expected,
                        pattern=pattern, strided=strided, native_compiled=not strided,
                        all_integer_outputs_exact=True, input_unchanged=True, RNG_unchanged=True,
                        changed_graph_inputs=bool(phase)))
                    save()
                if case in (2, 3):
                    with set_forward_context(None, proposer.vllm_config):
                        baseline_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(baseline_graph):
                            baseline_outputs = proposer.kernel(lengths, ids, mask)
                    graphs = dict(native_compiled=baseline_graph, fused=graph)
                    for trial in range(7):
                        for mode in (('native_compiled', 'fused') if trial%2 == 0 else ('fused', 'native_compiled')):
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(10): graphs[mode].replay()
                            end.record(); end.synchronize()
                            data['events'].append(dict(case=case, trial=trial, mode=mode, ms=start.elapsed_time(end)/10))
                    del baseline_outputs, baseline_graph, graphs
            data['medians_ms'] = {str(case): {mode: statistics.median(e['ms'] for e in data['events']
                if e['case']==case and e['mode']==mode) for mode in ('native_compiled', 'fused')} for case in (2, 3)}
        for path, sha in data['source_snapshots'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
        data.update(phase='complete', complete=True, source_scope='Full original GPU source plus compiled native kernel; strided integer64 native reference is direct eager method',
                    sample_quality_not_tested=True, service_throughput_not_tested=True)
    except BaseException as error:
        import traceback
        data.update(phase='failed', error=repr(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'], text=True))
        save()
    return 0 if data['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
