# SPDX-License-Identifier: Apache-2.0
"""Known direct-only leaf counts; isolate profiler compatibility from Graphs."""
import argparse
import json
from pathlib import Path
import signal
import torch
import vllm_fl
import triton
import triton.language as tl


@triton.jit
def tianshu_ixkn_coverage_a(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0)
    tl.store(Y + i, x + 1., i < N)


@triton.jit
def tianshu_ixkn_coverage_b(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0)
    tl.store(Y + i, x * 2., i < N)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-start", action="store_true")
    a = p.parse_args()
    signal.alarm(180)
    a.output.mkdir(parents=True, exist_ok=True)
    x = torch.arange(4096, device="cuda", dtype=torch.float32)
    y, z = torch.empty_like(x), torch.empty_like(x)

    def invoke():
        tianshu_ixkn_coverage_a[(16,)](x, y, 4096, 256)
        tianshu_ixkn_coverage_b[(16,)](y, z, 4096, 256)

    for _ in range(3):
        invoke()
    torch.cuda.synchronize()
    if not a.no_start:
        torch.cuda.cudart().cudaProfilerStart()
    invoke()
    torch.cuda.synchronize()
    if not a.no_start:
        torch.cuda.cudart().cudaProfilerStop()
    assert torch.equal(z, (x + 1) * 2)
    result = dict(complete=True, numeric_exact=True, profiler_start_called=not a.no_start,
                  expected_profiled_eager_leaves=0 if a.no_start else 2,
                  expected_profiled_graph_leaves=0,
                  expected_total_leaves=0 if a.no_start else 2,
                  unprofiled_warm_eager_leaves=6, unprofiled_warm_graph_leaves=0,
                  kernel_names=["tianshu_ixkn_coverage_a", "tianshu_ixkn_coverage_b"])
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
