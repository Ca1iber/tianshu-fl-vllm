# SPDX-License-Identifier: Apache-2.0
"""Standalone special-value math check; run only when the GPU is idle."""
import argparse
import json
import math
from pathlib import Path
import signal
import subprocess

hardware = subprocess.check_output(['ixsmi'], text=True)
assert 'No running processes found' in hardware
import torch
import vllm_fl
from vllm.triton_utils import tl, triton


@triton.jit
def special_values(x, exp_out, exp2_out, reciprocal_out, count: tl.constexpr):
    off = tl.arange(0, 16)
    value = tl.load(x + off, off < count, other=0.)
    tl.store(exp_out + off, tl.exp(value), off < count)
    tl.store(exp2_out + off, tl.exp2(value), off < count)
    tl.store(reciprocal_out + off, 1. / (tl.abs(value) + 1.), off < count)


def clean(values):
    return [value if math.isfinite(value) else str(value) for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'result.json').exists()
    signal.alarm(300)
    values = [float('-inf'), -1000., -100., -20., -1., 0., 1., 10., float('inf'), float('nan')]
    x = torch.tensor(values, device='cuda', dtype=torch.float32)
    results = [torch.empty_like(x) for _ in range(3)]
    special_values[(1,)](x, *results, len(values), num_warps=4)
    torch.cuda.synchronize()
    result = {'complete': True, 'official_benchmark': False, 'input': clean(values),
              'triton_exp': clean(results[0].cpu().tolist()),
              'triton_exp2': clean(results[1].cpu().tolist()),
              'triton_reciprocal': clean(results[2].cpu().tolist()),
              'torch_exp': clean(x.exp().cpu().tolist()),
              'torch_exp2': clean(x.exp2().cpu().tolist()),
              'cpu_exp': clean([math.exp(v) for v in values]),
              'hardware_before': hardware, 'torch': torch.__version__, 'triton': triton.__version__}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
