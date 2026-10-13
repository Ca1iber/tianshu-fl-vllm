# SPDX-License-Identifier: Apache-2.0
"""Check real IR dispatch, native rounding, functional aliases, and Graph output."""
import argparse
from datetime import datetime, timezone
import hashlib
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
    deadline = datetime.fromisoformat(json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())["stop_at_utc"])
    assert (deadline - datetime.now(timezone.utc)).total_seconds() > 900
    signal.alarm(900); root.mkdir(parents=True)
    import torch
    from vllm.ir.ops import layernorm as ops
    implementation = __import__("sys").modules["_tianshu_ir_rms_impl"]
    provider = implementation.PROVIDER
    assert provider in ops.rms_norm.impls and provider in ops.fused_add_rms_norm.impls
    assert os.environ["TIANSHU_IR_RMS_RECORD_DIR"] == str(root / "records")
    for path in [Path(__file__), Path(__file__).with_name("implementation.py"),
                 Path(__file__).with_name("hooks") / "sitecustomize.py"]:
        (root / ("sitecustomize.py" if path.name == "sitecustomize.py" else path.name)).write_bytes(path.read_bytes())
    data = dict(complete=False, numerical_checks_passed=False, official_benchmark=False, adopted=False,
                provider=provider, cases=[], source_sha256=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest())
    torch.manual_seed(2026101492)
    with torch.inference_mode(), ops.rms_norm.set_priority([provider, "native"]), ops.fused_add_rms_norm.set_priority([provider, "native"]):
        shapes = [(m, 2048, torch.bfloat16, w, None, False) for m in (1, 7, 64, 128, 576, 2048)
                  for w in (torch.bfloat16, torch.float32)]
        shapes += [(7, 1024, torch.bfloat16, torch.bfloat16, None, False),
                   (7, 2048, torch.float32, torch.float32, None, False),
                   (7, 2048, torch.bfloat16, None, None, False),
                   (7, 2048, torch.bfloat16, torch.bfloat16, 1024, False),
                   (7, 2048, torch.bfloat16, torch.bfloat16, None, True)]
        for m, n, dtype, wdtype, variance_size, strided in shapes:
            for distribution in ("random", "zero", "cancellation", "large", "small"):
                x = torch.randn(m, n * (2 if strided else 1), device="cuda", dtype=dtype)
                if strided:
                    x = x[:, ::2]
                residual = torch.randn_like(x)
                weight = torch.randn(n, device="cuda", dtype=wdtype) if wdtype is not None else None
                if distribution == "zero":
                    x.zero_(); residual.zero_()
                elif distribution == "cancellation":
                    residual = -x
                elif distribution == "large":
                    x.mul_(100); residual.mul_(100)
                elif distribution == "small":
                    x.mul_(.001); residual.mul_(.001)
                for kind in ("normal", "fused"):
                    op = ops.rms_norm if kind == "normal" else ops.fused_add_rms_norm
                    argv = (x, weight, 1e-5, variance_size) if kind == "normal" else (x, residual, weight, 1e-5, variance_size)
                    expected_provider = provider if op.impls[provider].supports_args(*argv) else "native"
                    assert op.dispatch(*argv).provider == expected_provider
                    saved_x = x.clone(); saved_r = residual.clone()
                    expected = op.impls["native"].impl_fn(*argv)
                    actual = op(*argv)
                    actual_y = actual[0] if kind == "fused" else actual
                    expected_y = expected[0] if kind == "fused" else expected
                    values = x.cpu().float() + residual.cpu().float() if kind == "fused" else x.cpu().float()
                    var_values = values if variance_size is None else values[:, :variance_size]
                    norm = values * torch.rsqrt(var_values.square().mean(-1, keepdim=True) + 1e-5)
                    reference = norm.to(weight.dtype) * weight.cpu() if weight is not None else norm
                    reference = reference.to(dtype)
                    assert torch.allclose(actual_y.cpu().float(), reference.float(), rtol=.016, atol=.016)
                    assert torch.allclose(actual_y.float(), expected_y.float(), rtol=.016, atol=.016)
                    assert torch.equal(x, saved_x) and torch.equal(residual, saved_r)
                    assert actual_y.data_ptr() != x.data_ptr()
                    if kind == "fused":
                        assert torch.equal(actual[1].cpu(), values.to(dtype))
                        assert actual[1].data_ptr() not in (x.data_ptr(), residual.data_ptr(), actual[0].data_ptr())
                        maybe = op.maybe_inplace(*argv)
                        assert torch.equal(maybe[0], actual[0]) and torch.equal(maybe[1], actual[1])
                        assert torch.equal(x, saved_x) and torch.equal(residual, saved_r)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_result = op(*argv)
                    graph.replay(); torch.cuda.synchronize()
                    gy = graph_result[0] if kind == "fused" else graph_result
                    assert torch.equal(gy, actual_y)
                    if kind == "fused":
                        assert torch.equal(graph_result[1], actual[1])
                    row = dict(rows=m, columns=n, activation_dtype=str(dtype), weight_dtype=str(wdtype),
                               distribution=distribution, variance_size=variance_size, strided=strided, kind=kind,
                               selected_provider=expected_provider, cpu_reference_passed=True,
                               native_reference_passed=True, functional_alias_passed=True, graph_passed=True,
                               exact_native=torch.equal(actual_y, expected_y))
                    data["cases"].append(row)
                    (root / "result.json").write_text(json.dumps(data, indent=2) + "\n")
    records = [json.loads(p.read_text()) for p in (root / "records").glob("*.json")]
    assert records and all(r["provider"] == provider and r["source_sha256"] == data["source_sha256"] for r in records)
    data.update(complete=True, numerical_checks_passed=True, actual_records=records,
                native_exact_cases=sum(r["exact_native"] for r in data["cases"]))
    (root / "result.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k not in ("cases", "actual_records")}))


if __name__ == "__main__":
    main()
