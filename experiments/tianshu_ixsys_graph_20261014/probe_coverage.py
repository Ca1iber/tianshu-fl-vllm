# SPDX-License-Identifier: Apache-2.0
"""Known-count CUDA Graph workload for auditing IXsys device-event coverage."""
import argparse
import json
from pathlib import Path
import signal
import torch
import triton
import triton.language as tl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "result.json").exists()
    signal.alarm(300)
    (root / "probe_coverage.py").write_bytes(Path(__file__).read_bytes())
    @triton.jit
    def tianshu_coverage_add(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(y + i, tl.load(x + i, i < N, other=0) + 1, i < N)

    @triton.jit
    def tianshu_coverage_mul(y, z, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(z + i, tl.load(y + i, i < N, other=0) * 2, i < N)

    with torch.inference_mode():
        x = torch.arange(4096, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x); z = torch.empty_like(x)

        def operations():
            a = tianshu_coverage_add[(16,)](x, y, 4096, 256, num_warps=4, num_stages=1)
            b = tianshu_coverage_mul[(16,)](y, z, 4096, 256, num_warps=4, num_stages=1)
            return a, b

        compiled = operations()
        for _ in range(5):
            operations()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operations()
        graph.replay(); torch.cuda.synchronize()
        expected = (x.cpu() + 1) * 2
        assert torch.equal(z.cpu(), expected)
        # IXsys --profile-from-start off should collect ONLY this range.
        torch.cuda.profiler.start()
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        assert torch.equal(z.cpu(), expected)
        data = dict(complete=True, official_benchmark=False, adopted=False,
                    scope="Profiler coverage probe, not model performance",
                    graph_replays_inside_profiler_range=20,
                    expected_device_kernel_calls=dict(tianshu_coverage_add=20, tianshu_coverage_mul=20),
                    graph_output_correct=True, profiler_coverage_verified=False,
                    compiled_kernel_names=[getattr(k, "name", None) for k in compiled],
                    torch_version=torch.__version__)
        (root / "result.json").write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(data), flush=True)


if __name__ == "__main__":
    main()
