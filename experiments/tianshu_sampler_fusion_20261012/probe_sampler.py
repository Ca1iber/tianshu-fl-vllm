# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU-oracle and alternating GPU timing probe; no model service."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess
import inspect
import importlib.metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'result.json').exists():
        raise RuntimeError('Refusing to overwrite a sampling probe')
    lease = Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    deadline = datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 0
    signal.alarm(min(1800, remaining))
    hardware = subprocess.check_output(['ixsmi'], text=True)
    assert 'No running processes found' in hardware, 'GPU probe must run serially'
    (root / 'hardware_before.txt').write_text(hardware)
    # Check ownership before compatibility imports can initialize our context.
    import torch
    import vllm_fl  # same installed Iluvatar compatibility initialization
    import flag_gems
    from experimental_sampler import prepare, sample_with_noise
    torch.set_num_threads(8)
    result = {'complete': False, 'official_benchmark': False,
              'installed_files_modified': False, 'adoption_allowed': False,
              'source_sha256': hashlib.sha256(Path(__file__).with_name('experimental_sampler.py').read_bytes()).hexdigest(),
              'cases': [], 'timings': [], 'started_utc': datetime.now(timezone.utc).isoformat()}
    result['packages']={name:importlib.metadata.version(name) for name in ('torch','flag-gems','vllm')}
    result['baseline_sources']={}
    for name in ('softmax','true_divide','argmax'):
        file=Path(inspect.getsourcefile(getattr(flag_gems,name)))
        result['baseline_sources'][name]={'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}

    def save():
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def baseline(x, q):
        return (x.softmax(-1, dtype=torch.float32) / q).argmax(-1)

    def gems_baseline(x, q):
        # Match production's FlagGems softmax / div / argmax dispatch.
        return flag_gems.argmax(flag_gems.true_divide(flag_gems.softmax(x,-1),q),-1)

    try:
        with torch.inference_mode():
            for batch in (1, 2, 7, 8, 16, 33, 64):
                for distribution in ('normal', 'peaked', 'uniform', 'masked', 'all_masked'):
                    seed = 2026101200 + batch
                    gen = torch.Generator(device='cuda').manual_seed(seed)
                    x = torch.randn((batch, 130560), device='cuda', generator=gen)
                    if distribution == 'peaked':
                        x.mul_(6)
                    elif distribution == 'uniform':
                        x.zero_()
                    elif distribution == 'masked':
                        x[:, 4096:] = float('-inf')
                    elif distribution == 'all_masked':
                        x.fill_(float('-inf'))
                    q = torch.empty_like(x).exponential_(generator=gen)
                    # CPU FP64 exponential-race oracle, independent of GPU kernels.
                    xc, qc = x.double().cpu(), q.double().cpu()
                    expected = (xc - qc.log()).argmax(-1)
                    reference = baseline(x, q).cpu()
                    production = gems_baseline(x,q).cpu()
                    for block in (1024, 2048, 4096):
                        got = sample_with_noise(x, q, prepare(x, block), block).cpu()
                        item = {'batch': batch, 'distribution': distribution, 'block': block,
                                'baseline_equal': torch.equal(got, reference),
                                'flaggems_baseline_equal':torch.equal(got,production),
                                'cpu_oracle_equal': torch.equal(got, expected),
                                'candidate': got.tolist(), 'reference': reference.tolist(),
                                'flaggems_reference':production.tolist(),'oracle': expected.tolist()}
                        result['cases'].append(item)
                        print(json.dumps(item), flush=True)
                    save()
                for block in (1024, 2048, 4096):
                    scratch = prepare(x, block)
                    # Real finite logits for performance; validation used masks above.
                    x.normal_()
                    for _ in range(10):
                        baseline(x, q)
                        gems_baseline(x,q)
                        sample_with_noise(x, q, scratch, block)
                    torch.cuda.synchronize()
                    graphs=[]
                    for function in (lambda:baseline(x,q),lambda:gems_baseline(x,q),
                                     lambda:sample_with_noise(x,q,scratch,block)):
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(30):function()
                        graphs.append(graph)
                    measurements = [[], [], []]
                    for repetition in range(7):
                        order=[0,1,2]
                        for selected in order[repetition%3:]+order[:repetition%3]:
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            graphs[selected].replay()
                            end.record()
                            end.synchronize()
                            measurements[selected].append(start.elapsed_time(end) / 30)
                    native,before,after=map(statistics.median,measurements)
                    item = {'batch': batch, 'block': block, 'torch_native_ms':native,
                            'flaggems_baseline_ms': before, 'candidate_ms': after,
                            'change_percent': 100 * (after / before - 1), 'raw_ms': measurements,
                            'includes_noise_generation': False, 'timing_mode': 'graph_replay_events'}
                    result['timings'].append(item)
                    print(json.dumps(item), flush=True)
                    save()
            result.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                          cases_passed=sum(c['baseline_equal'] and c['flaggems_baseline_equal'] and c['cpu_oracle_equal'] for c in result['cases']),
                          cases_total=len(result['cases']))
    finally:
        save()


if __name__ == '__main__':
    main()
