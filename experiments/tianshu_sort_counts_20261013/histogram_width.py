# SPDX-License-Identifier: Apache-2.0
"""Process-only radix histogram accumulator variants, preserving sort indices."""
import hashlib
import importlib.util
from pathlib import Path
import sys


def safe_width(requested, tiles_per_cta, n, tile_n):
    assert requested in (8, 16, 32, 64)
    assert tiles_per_cta > 0 and tile_n > 0 and 0 < n < 2**30
    bound = min(tiles_per_cta, (n + tile_n - 1) // tile_n)
    for width in (8, 16, 32, 64):
        if width >= requested and bound <= 2**(width - 1) - 1:
            return width, bound
    raise AssertionError('No safe accumulator width')


def create_kernels(original_module, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    text = Path(original_module.__file__).read_text()
    start = text.index('@triton.jit\ndef compute_global_hist_kernel(')
    end = text.index('\n\n@triton.jit\ndef sweep(', start)
    body = text[start:end]
    assert body.count('acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int64)') == 1
    assert body.count('local_sum = tl.sum(acc, axis=1)') == 1
    kernels, provenance = {}, {}
    for width in (64, 32, 16, 8):
        candidate = body.replace('dtype=tl.int64)', f'dtype=tl.int{width})')
        if width < 32:
            candidate = candidate.replace('local_sum = tl.sum(acc, axis=1)',
                                          'local_sum = tl.sum(acc, axis=1, dtype=tl.int32)')
        source = ('# SPDX-License-Identifier: Apache-2.0\n'
                  'import triton\nimport triton.language as tl\n'
                  'from flag_gems.ops.sort import convert_to_uint_preverse_order\n\n' + candidate + '\n')
        path = directory / f'count{width}.py'
        if path.exists():
            assert path.read_text() == source
        else:
            path.write_text(source)
        name = f'_tianshu_histogram_count{width}_{hashlib.sha256(source.encode()).hexdigest()[:12]}'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        kernels[width] = module.compute_global_hist_kernel
        provenance[width] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 accumulator_bits=width, reduction_bits=32 if width < 32 else width)
    return kernels, provenance


class HistogramRouter:
    def __init__(self, original, kernels, requested, record=None):
        self.original = original
        self.kernels = kernels
        self.requested = requested
        self.record = record

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            # Keep the existing public kernel call unchanged. Fall back when
            # the host does not provide the inspected radix-sort geometry.
            if len(args) < 9 or not all(isinstance(args[i], int) for i in (4, 5, 6)):
                return self.original[grid](*args, **kwargs)
            n, tiles, tile_n = args[4:7]
            if not (0 < n < 2**30 and tiles > 0 and tile_n > 0):
                return self.original[grid](*args, **kwargs)
            actual, bound = safe_width(self.requested, tiles, n, tile_n)
            result = self.kernels[actual][grid](*args, **kwargs)
            if self.record:
                self.record(dict(requested_bits=self.requested, actual_bits=actual,
                                 per_lane_count_bound=bound, n=n,
                                 tile_n=tile_n, tiles_per_cta=tiles))
            return result
        return invoke
