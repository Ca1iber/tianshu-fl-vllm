# SPDX-License-Identifier: Apache-2.0
"""Exact CPU histograms/stable sort plus graph timings for bounded counters."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import signal
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 0
    signal.alarm(min(3600, remaining))
    hardware = subprocess.check_output(['ixsmi'], text=True)
    assert 'No running processes found' in hardware
    (root / 'hardware_before.txt').write_text(hardware)
    import numpy as np
    import torch
    import triton
    import flag_gems
    from histogram_width import create_kernels, HistogramRouter, safe_width
    module = importlib.import_module('flag_gems.ops.sort')
    original = module.compute_global_hist_kernel
    source = Path(module.__file__)
    kernels, generated = create_kernels(module, root / 'generated')
    snapshot = root / 'source_snapshot'
    snapshot.mkdir()
    for path in (source, Path(__file__), Path(__file__).with_name('histogram_width.py')):
        (snapshot / path.name).write_bytes(path.read_bytes())
    torch.set_num_threads(8)
    data = dict(complete=False, official_benchmark=False, adoption_allowed=False,
                original_sort_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                generated=generated, histogram_cases=[], sort_cases=[],
                started_utc=datetime.now(timezone.utc).isoformat(), installed_sources_modified=False)

    def save():
        p = root / 'result.tmp'
        p.write_text(json.dumps(data, indent=2) + '\n')
        p.replace(root / 'result.json')

    def cpu_hist(x, bits, descending):
        width = x.element_size() * 8
        unsigned = {16: np.uint16, 32: np.uint32}[width]
        raw = x.cpu().view(torch.int16 if width == 16 else torch.int32).numpy().view(unsigned)
        sign = unsigned(1 << (width - 1))
        full = unsigned((1 << width) - 1)
        key = raw ^ np.where((raw & sign) != 0, full, sign).astype(unsigned)
        if descending:
            key = key ^ full
        passes = (width + bits - 1) // bits
        result = np.empty((x.shape[0], passes, 1 << bits), dtype=np.int32)
        for row in range(x.shape[0]):
            for p in range(passes):
                digit = ((key[row] >> (p * bits)) & unsigned((1 << bits) - 1)).astype(np.int64)
                result[row, p] = np.bincount(digit, minlength=1 << bits)
        return torch.from_numpy(result)

    save()
    scenarios = [(1, 8191, 'random', torch.float32, 8),
                 (2, 8193, 'ties', torch.float32, 8),
                 (64, 130560, 'random', torch.float32, 8),
                 (64, 130560, 'zeros', torch.float32, 8),
                 (2, 130560, 'masked', torch.bfloat16, 8),
                 (2, 8193, 'random', torch.float16, 8),
                 (1, 130048, 'zeros', torch.float32, 127),
                 (1, 131072, 'zeros', torch.float32, 128)]
    with torch.inference_mode():
        for case, (m, n, pattern, dtype, tiles) in enumerate(scenarios):
            torch.manual_seed(2026101300 + case)
            if pattern == 'zeros':
                x = torch.zeros((m, n), dtype=dtype, device='cuda')
            elif pattern == 'ties':
                x = torch.randint(-8, 8, (m, n), device='cuda').to(dtype)
            else:
                x = torch.randn((m, n), dtype=dtype, device='cuda')
                if pattern == 'masked':
                    x[:, ::4] = -float('inf')
            for descending in (False, True):
                bits = 4
                passes = x.element_size() * 8 // bits
                grid = (m * triton.cdiv(n, 1024 * tiles), 1, 1)
                expected = cpu_hist(x, bits, descending)
                case_row = dict(m=m, n=n, pattern=pattern, dtype=str(dtype), descending=descending,
                                tiles_per_cta=tiles, variants=[])
                graphs, outputs = [], []
                variants = [('original64', original, 64)] + [(str(w), kernels[w], w) for w in (64, 32, 16, 8)]
                for label, kernel, requested in variants:
                    actual, bound = safe_width(requested, tiles, n, 1024)
                    selected = original if label == 'original64' else kernels[actual]
                    output = torch.empty((m, passes, 16), dtype=torch.int32, device='cuda')
                    row = dict(label=label, requested_bits=requested, actual_bits=actual,
                               per_lane_bound=bound)
                    try:
                        def operation():
                            output.zero_()
                            return selected[grid](x, output, passes, m, n, tiles, 1024, 16, bits, descending)
                        for _ in range(3):
                            compiled = operation()
                        torch.cuda.synchronize()
                        assert torch.equal(output.cpu(), expected), 'Histogram differs from exact CPU bit-key counts'
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            operation()
                        graph.replay()
                        torch.cuda.synchronize()
                        assert torch.equal(output.cpu(), expected)
                        row['exact_cpu_histogram_passed'] = True
                        row['metadata'] = {k: getattr(getattr(compiled, 'metadata', None), k, None)
                                           for k in ('num_warps', 'num_stages', 'shared')}
                        row['registers'] = getattr(compiled, 'n_regs', None)
                        row['spills'] = getattr(compiled, 'n_spills', None)
                        for kind in ('ttir', 'ttgir', 'llir', 'ptx'):
                            text = getattr(compiled, 'asm', {}).get(kind)
                            if isinstance(text, str):
                                (root / f'case{case}_{int(descending)}_{label}.{kind}.txt').write_text(text)
                        graphs.append((row, graph))
                        outputs.append(output)
                    except Exception as error:
                        row.update(exact_cpu_histogram_passed=False, error=str(error))
                    case_row['variants'].append(row)
                timings = {id(row): [] for row, graph in graphs}
                for repeat in range(7):
                    order = graphs[repeat % len(graphs):] + graphs[:repeat % len(graphs)] if graphs else []
                    for row, graph in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        timings[id(row)].append(start.elapsed_time(end) / 20)
                for row, graph in graphs:
                    row['raw_ms'] = timings[id(row)]
                    row['median_ms'] = statistics.median(row['raw_ms'])
                data['histogram_cases'].append(case_row)
                save()
                print(json.dumps(dict(hist_case=case, descending=descending,
                                      variants=[{k: v for k, v in row.items() if k != 'raw_ms'}
                                                for row in case_row['variants']])), flush=True)
                del graphs, outputs, x
                x = None
                # Recreate once below only for the other direction; tensors
                # stay outside the timings and use the exact same seed.
                if not descending:
                    torch.manual_seed(2026101300 + case)
                    if pattern == 'zeros':
                        x = torch.zeros((m, n), dtype=dtype, device='cuda')
                    elif pattern == 'ties':
                        x = torch.randint(-8, 8, (m, n), device='cuda').to(dtype)
                    else:
                        x = torch.randn((m, n), dtype=dtype, device='cuda')
                        if pattern == 'masked':
                            x[:, ::4] = -float('inf')
            torch.cuda.empty_cache()
        for case, (m, n) in enumerate([(1, 8191), (2, 8193), (1, 130560),
                                      (32, 130560), (64, 130560)]):
            for pattern in ('random', 'ties', 'masked'):
                torch.manual_seed(2026101400 + case)
                x = torch.randint(-8, 8, (m, n), device='cuda').float() if pattern == 'ties' else torch.randn((m, n), device='cuda')
                if pattern == 'masked':
                    x[:, ::4] = -float('inf')
                for descending in (False, True):
                    cpu_values, cpu_indices = torch.sort(x.cpu(), dim=-1, stable=True, descending=descending)
                    sort_graphs = []
                    for width in (64, 32, 16, 8):
                        row = dict(m=m, n=n, pattern=pattern, descending=descending, bits=width)
                        records = []
                        try:
                            module.compute_global_hist_kernel = HistogramRouter(original, kernels, width, records.append)
                            with flag_gems.use_gems():
                                values, indices = torch.sort(x, dim=-1, stable=True, descending=descending)
                            row.update(actual_histogram_routed=bool(records), values_exact=torch.equal(values.cpu(), cpu_values),
                                       stable_indices_exact=torch.equal(indices.cpu(), cpu_indices),
                                       indices_dtype=str(indices.dtype), routing=records)
                            assert records and row['values_exact'] and row['stable_indices_exact']
                            assert indices.dtype == torch.int64
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                with flag_gems.use_gems():
                                    graph_values, graph_indices = torch.sort(
                                        x, dim=-1, stable=True, descending=descending)
                            graph.replay()
                            torch.cuda.synchronize()
                            row['graph_values_exact'] = torch.equal(graph_values.cpu(), cpu_values)
                            row['graph_stable_indices_exact'] = torch.equal(graph_indices.cpu(), cpu_indices)
                            assert row['graph_values_exact'] and row['graph_stable_indices_exact']
                            sort_graphs.append((row, graph, graph_values, graph_indices))
                        except Exception as error:
                            row['error'] = str(error)
                        finally:
                            module.compute_global_hist_kernel = original
                        data['sort_cases'].append(row)
                        save()
                    timings = {id(row): [] for row, *_ in sort_graphs}
                    for repeat in range(7):
                        order = (sort_graphs[repeat % len(sort_graphs):] +
                                 sort_graphs[:repeat % len(sort_graphs)]) if sort_graphs else []
                        for row, graph, *_ in order:
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(10):
                                graph.replay()
                            end.record()
                            end.synchronize()
                            timings[id(row)].append(start.elapsed_time(end) / 10)
                    for row, *_ in sort_graphs:
                        row['raw_sort_ms'] = timings[id(row)]
                        row['median_sort_ms'] = statistics.median(row['raw_sort_ms'])
                    save()
                    del sort_graphs
    data.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == '__main__':
    main()
