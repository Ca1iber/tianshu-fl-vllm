# SPDX-License-Identifier: Apache-2.0
"""Numerical, RNG-state, distribution and full eager-cost probe; no service."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import signal
import statistics
import subprocess
import time


def philox_cpu(seed, counter):
    """Independent Random123 Philox4x32-10 integer reference."""
    mask = (1 << 32) - 1
    c0, c1, c2, c3 = counter & mask, (counter >> 32) & mask, 0, 0
    k0, k1 = seed & mask, (seed >> 32) & mask
    for _ in range(10):
        p0, p1 = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & mask, p1 & mask, ((p0 >> 32) ^ c3 ^ k1) & mask, p0 & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return [c0, c1, c2, c3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True)
    signal.alarm(1500)
    hardware = subprocess.check_output(['ixsmi'], text=True)
    assert 'No running processes found' in hardware
    (root / 'hardware_before.txt').write_text(hardware)
    import torch
    import vllm_fl
    import flag_gems
    import experimental_rng_sampler as candidate
    torch.set_num_threads(8)
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('previous_fixed_noise_sampler', repo / 'experiments/tianshu_sampler_fusion_20261012/experimental_sampler.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    result = dict(complete=False, official_benchmark=False, installed_files_modified=False,
                  adoption_allowed=False, new_global_random_sequence=True,
                  per_request_generators_must_fall_back=True,
                  captures_must_fall_back=True, rng_words=[], cases=[], timings=[], state_checks=[],
                  distribution_checks=[], started_utc=datetime.now(timezone.utc).isoformat())
    result['sources'] = {}
    paths = [Path(__file__).resolve(), Path(candidate.__file__), Path(old.__file__),
             Path(inspect.getsourcefile(flag_gems.exponential_)),
             Path(inspect.getsourcefile(candidate.philox_backend_seed_offset))]
    for path in paths:
        data = path.read_bytes()
        result['sources'][str(path)] = hashlib.sha256(data).hexdigest()
        target = root / 'source_snapshot' / path.relative_to('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def save():
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def production(x):
        q = torch.empty_like(x)
        flag_gems.exponential_(q)
        return flag_gems.argmax(flag_gems.true_divide(flag_gems.softmax(x, -1), q), -1)

    def old_fused(x):
        q = torch.empty_like(x)
        flag_gems.exponential_(q)
        return old.sample_with_noise(x, q, old.prepare(x))

    try:
        assert philox_cpu(0, 0) == [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
        with torch.inference_mode():
            for seed in (0, 12345, (1 << 63) + 17):
                for offset in (0, (1 << 32) - 8, (1 << 32) + 12, (1 << 48) + 4):
                    got = candidate.random_words(seed, offset, 17).cpu().tolist()
                    expected = [philox_cpu(seed, offset+i) for i in range(17)]
                    passed = all([got[j][i] for j in range(4)] == expected[i] for i in range(17))
                    item = dict(seed=seed, offset=offset, counters=17, exact=passed)
                    result['rng_words'].append(item)
                    save()
                    assert passed, item
            cpu_gen = torch.Generator().manual_seed(20261016)
            for batch, vocab in ((1, 130560), (7, 130560), (8, 130560), (64, 130560), (3, 129999)):
                cpu_logits = torch.randn((batch, vocab), generator=cpu_gen)
                for distribution in ('normal', 'peaked', 'uniform', 'masked', 'all_masked', 'nan', 'positive_inf'):
                    host = cpu_logits.clone()
                    if distribution == 'peaked': host.mul_(6)
                    if distribution == 'uniform': host.zero_()
                    if distribution == 'masked': host[:, 4096:] = -float('inf')
                    if distribution == 'all_masked': host.fill_(-float('inf'))
                    if distribution == 'nan': host[:, 19] = float('nan')
                    if distribution == 'positive_inf': host[:, 19] = float('inf')
                    x = host.cuda()
                    for block in (1024, 2048, 4096):
                        noise = torch.empty_like(x)
                        offset = (1 << 32) - 4
                        token = candidate.sample_seeded(x, candidate.prepare(x, block), 997, offset, block, noise=noise).clone()
                        fused = candidate.sample_seeded(x, candidate.prepare(x, block), 997, offset, block).cpu()
                        q = noise.cpu().double()
                        # Build the FP64 oracle after transfer to CPU, never on this GPU.
                        expected = (host.double().softmax(-1) / q).argmax(-1)
                        baseline = flag_gems.argmax(flag_gems.true_divide(flag_gems.softmax(x, -1), noise), -1).cpu()
                        item = dict(batch=batch, vocab=vocab, distribution=distribution, block=block,
                                    materialized_matches_fused=torch.equal(token.cpu(), fused),
                                    cpu_reference_equal=torch.equal(expected, fused),
                                    production_same_noise_equal=torch.equal(baseline, fused),
                                    candidate=fused.tolist(), cpu_reference=expected.tolist())
                        result['cases'].append(item)
                        save()
                        assert item['materialized_matches_fused'] and item['cpu_reference_equal'] and item['production_same_noise_equal'], item
            for batch in (1, 8, 64):
                x = torch.randn((batch, 130560), generator=cpu_gen).cuda()
                assert candidate.eligible(x, {})
                assert not candidate.eligible(x, {0: torch.Generator(device='cuda')})
                assert not candidate.eligible(x, {}, True)
                generator = torch.cuda.default_generators[0]
                generator.manual_seed(12345)
                generator.set_offset((1 << 32) - 4)
                before = generator.get_offset()
                first = candidate.sample_global(x).clone()
                middle = generator.get_offset()
                second = candidate.sample_global(x).clone()
                after = generator.get_offset()
                generator.manual_seed(12345)
                generator.set_offset(before)
                replay = candidate.sample_global(x).clone()
                item = dict(batch=batch, offset_before=before, offset_middle=middle, offset_after=after,
                            reservation=candidate.reservation_size(x), exact_advance=(middle-before == after-middle == candidate.reservation_size(x)),
                            reset_reproducible=torch.equal(first, replay), successive_draws_differ=not torch.equal(first, second))
                result['state_checks'].append(item)
                save()
                assert item['exact_advance'] and item['reset_reproducible']
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    capture_eligible = candidate.eligible(x, {})
                    graph_token = candidate.sample_seeded(x, candidate.prepare(x), 997, 1234)
                assert not capture_eligible
                graph.replay()
                expected = candidate.sample_seeded(x, candidate.prepare(x), 997, 1234).clone()
                assert torch.equal(graph_token, expected)
                functions = (lambda: production(x), lambda: old_fused(x), lambda: candidate.sample_global(x))
                for function in functions:
                    for _ in range(5): function()
                torch.cuda.synchronize()
                measurements = [[], [], []]
                gpu_measurements = [[], [], []]
                for repeat in range(7):
                    order = list(range(3)); order = order[repeat % 3:] + order[:repeat % 3]
                    for selected in order:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        ev0.record()
                        for _ in range(20): functions[selected]()
                        ev1.record(); ev1.synchronize()
                        measurements[selected].append(1000 * (time.perf_counter() - start) / 20)
                        gpu_measurements[selected].append(ev0.elapsed_time(ev1) / 20)
                medians = list(map(statistics.median, measurements))
                item = dict(batch=batch, block=2048, mode='actual eager calls; RNG state and all allocations included; no TopP',
                            production_ms=medians[0], old_fused_ms=medians[1], candidate_ms=medians[2],
                            change_percent=100*(medians[2]/medians[0]-1), raw_wall_ms=measurements, raw_event_ms=gpu_measurements,
                            capture_guard_passed=True, fixed_counter_graph_output_passed=True)
                result['timings'].append(item); save(); print(json.dumps(item), flush=True)
            for seed in (773, 886):
                rows, vocab = 16384, 16
                logits = torch.arange(1, vocab+1, dtype=torch.float64).log().float()
                x = logits.repeat(rows, 1).cuda()
                got = candidate.sample_seeded(x, candidate.prepare(x), seed, (1 << 32)-4).cpu()
                counts = torch.bincount(got, minlength=vocab).double()
                probability = torch.arange(1, vocab+1, dtype=torch.float64); probability /= probability.sum()
                expected = rows*probability
                tolerance = 6*(rows*probability*(1-probability)).sqrt()+10
                passed = bool(torch.all((counts-expected).abs() <= tolerance))
                item = dict(seed=seed, draws=rows, vocab=vocab, counts=counts.tolist(), expected=expected.tolist(), tolerance=tolerance.tolist(), passed=passed)
                result['distribution_checks'].append(item); save(); assert passed, item
            result.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
