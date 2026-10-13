# SPDX-License-Identifier: Apache-2.0
"""Same-body RMSNorm launch sweep; graph timings include identical input resets."""
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    lease = json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())
    assert (datetime.fromisoformat(lease["stop_at_utc"]) - datetime.now(timezone.utc)).total_seconds() > 600
    signal.alarm(600)
    root.mkdir(parents=True)
    (root / "probe_launch.py").write_bytes(Path(__file__).read_bytes())
    import torch
    import triton
    from flag_gems.config import use_c_extension
    normal = importlib.import_module("flag_gems.ops.rms_norm")
    fused = importlib.import_module("flag_gems.fused.fused_add_rms_norm")
    data = dict(complete=False, official_benchmark=False, adopted=False, cases=[],
                dtype="bfloat16", columns=2048, timing_includes_input_reset=True,
                use_c_extension=use_c_extension, kernel_bodies_unchanged=True,
                source_hashes={}, errors=[])
    for module in (normal, fused):
        p = Path(module.__file__)
        raw = p.read_bytes()
        (root / p.name).write_bytes(raw)
        data["source_hashes"][module.__name__] = hashlib.sha256(raw).hexdigest()
    assert data["source_hashes"] == {
        "flag_gems.ops.rms_norm": "58407466be597c05442b3288c95abb2c33e47f0f6b98cff624b2f2e8a00823e3",
        "flag_gems.fused.fused_add_rms_norm": "192cbddc2283a564a9eaf90bfae1641a8a052398c0d7dd14f83f6a67f76ec3b9"}
    plans = [("default", {}), ("w1s1", dict(num_warps=1, num_stages=1)),
             ("w2s1", dict(num_warps=2, num_stages=1)), ("w4s1", dict(num_warps=4, num_stages=1)),
             ("w8s1", dict(num_warps=8, num_stages=1))]
    # LibEntry keys omit explicit launch kwargs; use its identical raw JIT
    # body so each warp configuration has a distinct Triton compile key.
    normal_kernel = normal.rms_norm_kernel.fn
    fused_kernel = fused.fused_add_rms_norm_kernel.fn
    assert isinstance(normal_kernel, triton.runtime.JITFunction)
    assert isinstance(fused_kernel, triton.runtime.JITFunction)

    def save():
        (root / "result.json").write_text(json.dumps(data, indent=2) + "\n")

    with torch.inference_mode():
        torch.manual_seed(2026101488)
        for rows in (1, 4, 8, 16, 32, 64, 128, 576, 2048):
            x0 = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
            r0 = torch.randn_like(x0)
            w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
            xc = x0.cpu().float()
            rc = r0.cpu().float()
            wc = w.cpu().float()
            for kind in ("normal", "fused"):
                summed = xc + rc if kind == "fused" else xc
                variance = (summed * summed).mean(-1, keepdim=True)
                normalized = summed * torch.rsqrt(variance + 1e-5)
                reference = ((normalized * wc).bfloat16() if kind == "fused"
                             else (normalized.bfloat16() * wc.bfloat16()))
                case = dict(rows=rows, kind=kind, variants=[])
                graphs = []
                baseline_output = baseline_residual = None
                for label, config in plans:
                    try:
                        x = x0.clone(); residual = r0.clone()
                        output = torch.empty_like(x)
                        inverse = torch.empty(rows, device="cuda", dtype=torch.float32)

                        def operation():
                            if kind == "fused":
                                x.copy_(x0); residual.copy_(r0)
                                return fused_kernel[(rows,)](
                                    x, residual, w, 2048, 1, 2048, 1, 2048, 1e-5, 2048, **config)
                            return normal_kernel[(rows,)](
                                output, inverse, x, w, 2048, 1, 2048, 1, 2048, 1e-5, 2048, **config)

                        compiled = operation()
                        if isinstance(compiled, tuple):
                            compiled = compiled[0]
                        if config:
                            assert compiled.metadata.num_warps == config["num_warps"]
                            assert compiled.metadata.num_stages == config["num_stages"]
                        torch.cuda.synchronize()
                        actual = (x if kind == "fused" else output).cpu()
                        actual_residual = residual.cpu() if kind == "fused" else None
                        if label == "default":
                            baseline_output = actual.clone()
                            baseline_residual = actual_residual.clone() if actual_residual is not None else None
                        exact = torch.equal(actual, baseline_output)
                        residual_equal = (actual_residual is None or torch.equal(actual_residual, baseline_residual))
                        numeric = torch.allclose(actual.float(), reference.float(), rtol=.016, atol=.016)
                        if actual_residual is not None:
                            numeric = numeric and torch.equal(actual_residual, summed.bfloat16())
                        v = dict(label=label, configuration=config, exact_equal_default=exact,
                                 residual_equal_default=residual_equal, cpu_fp32_reference_passed=bool(numeric),
                                 max_abs_error_cpu_fp32=float((actual.float() - reference.float()).abs().max()),
                                 compiled_metadata={k: getattr(compiled.metadata, k, None)
                                                    for k in ("num_warps", "num_stages", "shared")}, samples_ms=[])
                        case["variants"].append(v)
                        if not numeric:
                            continue
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(20):
                                operation()
                        graph.replay()
                        torch.cuda.synchronize()
                        assert torch.equal((x if kind == "fused" else output).cpu(), actual)
                        if actual_residual is not None:
                            assert torch.equal(residual.cpu(), actual_residual)
                        v["graph_replay_matches_eager"] = True
                        graphs.append((v, graph, (x, residual, output, inverse)))
                    except Exception as error:
                        case["variants"].append(dict(label=label, error=repr(error)))
                for rotation in range(7):
                    order = graphs[rotation % len(graphs):] + graphs[:rotation % len(graphs)] if graphs else []
                    for v, graph, keepalive in order:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):
                            graph.replay()
                        end.record(); end.synchronize()
                        v["samples_ms"].append(start.elapsed_time(end) / 400)
                for v in case["variants"]:
                    if v.get("samples_ms"):
                        v["median_ms"] = statistics.median(v["samples_ms"])
                data["cases"].append(case)
                save()
                print(json.dumps(dict(rows=rows, kind=kind, variants=[
                    {k: v.get(k) for k in ("label", "median_ms", "exact_equal_default", "cpu_fp32_reference_passed", "error")}
                    for v in case["variants"]])), flush=True)
    data["complete"] = True
    save()


if __name__ == "__main__":
    main()
