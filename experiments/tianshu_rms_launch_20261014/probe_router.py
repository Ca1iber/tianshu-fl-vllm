# SPDX-License-Identifier: Apache-2.0
"""Check the actual small-RMS router on numeric boundaries and graph replay."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
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
    import torch
    from flag_gems.modules.normalization import gems_rms_forward
    normal = importlib.import_module("flag_gems.ops.rms_norm")
    fused = importlib.import_module("flag_gems.fused.fused_add_rms_norm")
    hook = Path(__file__).with_name("hooks") / "sitecustomize.py"
    assert os.environ["TIANSHU_RMS_LAUNCH_PROBE"] == "1"
    assert Path(os.environ["TIANSHU_RMS_LAUNCH_RECORD_DIR"]) == root / "rms_records"
    (root / "probe_router.py").write_bytes(Path(__file__).read_bytes())
    (root / "rms_sitecustomize.py").write_bytes(hook.read_bytes())
    routers = [(normal, "rms_norm_kernel", normal.rms_norm_kernel),
               (fused, "fused_add_rms_norm_kernel", fused.fused_add_rms_norm_kernel)]
    assert all(hasattr(m, "_tianshu_rms_original_kernel") for m, _, _ in routers)
    data = dict(complete=False, numerical_checks_passed=False, official_benchmark=False,
                adopted=False, cases=[], hook_sha256=hashlib.sha256(hook.read_bytes()).hexdigest())
    torch.manual_seed(2026101491)
    with torch.inference_mode():
        shapes = [(m, 2048, torch.bfloat16) for m in (1, 7, 16, 64, 65, 128)]
        shapes += [(7, 1024, torch.bfloat16), (7, 2048, torch.float32)]
        for rows, n, dtype in shapes:
            for distribution in ("random", "zero", "cancellation", "large", "small"):
                x0 = torch.randn(rows, n, device="cuda", dtype=dtype)
                r0 = torch.randn_like(x0)
                w = torch.randn(n, device="cuda", dtype=dtype)
                if distribution == "zero":
                    x0.zero_(); r0.zero_()
                elif distribution == "cancellation":
                    r0 = -x0
                elif distribution == "large":
                    x0.mul_(100); r0.mul_(100)
                elif distribution == "small":
                    x0.mul_(.001); r0.mul_(.001)
                for kind in ("normal", "fused"):
                    x = x0.clone(); residual = r0.clone() if kind == "fused" else None
                    for module, name, _ in routers:
                        setattr(module, name, module._tianshu_rms_original_kernel)
                    expected = gems_rms_forward(x, residual, w, 1e-5)
                    expected = tuple(t.clone() for t in expected) if isinstance(expected, tuple) else expected.clone()
                    for module, name, router in routers:
                        setattr(module, name, router)
                    x = x0.clone(); residual = r0.clone() if kind == "fused" else None
                    actual = gems_rms_forward(x, residual, w, 1e-5)
                    actual_y = actual[0] if isinstance(actual, tuple) else actual
                    expected_y = expected[0] if isinstance(expected, tuple) else expected
                    cpu = x0.cpu().float() + r0.cpu().float() if kind == "fused" else x0.cpu().float()
                    cpu_norm = cpu * torch.rsqrt((cpu * cpu).mean(-1, keepdim=True) + 1e-5)
                    cpu_ref = ((cpu_norm * w.cpu().float()).to(dtype) if kind == "fused"
                               else cpu_norm.to(dtype) * w.cpu())
                    checked = torch.allclose(actual_y.cpu().float(), cpu_ref.float(), rtol=.016, atol=.016)
                    if kind == "fused":
                        checked = checked and torch.equal(actual[1].cpu(), cpu.to(dtype))
                    assert checked, (rows, n, dtype, distribution, kind)
                    saved_y = actual_y.clone()
                    saved_r = residual.clone() if residual is not None else None
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        x.copy_(x0)
                        if residual is not None:
                            residual.copy_(r0)
                        graph_y = gems_rms_forward(x, residual, w, 1e-5)
                    graph.replay(); torch.cuda.synchronize()
                    gy = graph_y[0] if isinstance(graph_y, tuple) else graph_y
                    assert torch.equal(gy, saved_y)
                    if saved_r is not None:
                        assert torch.equal(residual, saved_r)
                    data["cases"].append(dict(rows=rows, columns=n, dtype=str(dtype), distribution=distribution,
                                              kind=kind, cpu_fp32_reference_passed=True,
                                              graph_matches_eager=True, exact_equal_original=torch.equal(actual_y, expected_y)))
                    (root / "result.json").write_text(json.dumps(data, indent=2) + "\n")
    records = [json.loads(p.read_text()) for p in (root / "rms_records").glob("*.json")]
    assert records and {(x["kind"], x["rows"]) for x in records} == {(k, m) for k in ("normal", "fused") for m in (1, 7, 16, 64)}
    assert all(r["configuration"] == dict(num_warps=8, num_stages=1) for r in records)
    data.update(complete=True, numerical_checks_passed=True, actual_routing_records=records,
                original_equal_cases=sum(x["exact_equal_original"] for x in data["cases"]))
    (root / "result.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k not in ("cases", "actual_routing_records")}))


if __name__ == "__main__":
    main()
