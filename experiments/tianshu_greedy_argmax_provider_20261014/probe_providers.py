# SPDX-License-Identifier: Apache-2.0
"""Exact CPU/native greedy semantics and paired aggregate Graph timings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.output.resolve()
    assert not root.exists()
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    root.mkdir(parents=True)
    stop = datetime.fromisoformat(json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())["stop_at_utc"])
    remaining = int((stop - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 0
    signal.alarm(min(1200, remaining))
    import torch
    import vllm_fl
    from vllm.v1.sample.sampler import Sampler
    import flag_gems
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "tianshu_greedy_argmax_20261014"))
    from implementation import launch
    torch.set_num_threads(8)
    base = Path(__file__).resolve().parent
    source = Path("/usr/local/lib/python3.12/site-packages/vllm/v1/sample/sampler.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "1a2f18ddc91c89f3744ab5a93624be056bf20b7a8b2e6e13d0236182718a2253"
    snapshots = root / "source_snapshot"
    snapshots.mkdir()
    for path in (Path(__file__), base.parent / "tianshu_greedy_argmax_20261014/implementation.py", source, Path(flag_gems.argmax.__code__.co_filename)):
        (snapshots / path.name).write_bytes(path.read_bytes())
    plans = [dict(provider=name) for name in ("explicit_flag_gems", "custom")]
    data = dict(complete=False, official_benchmark=False, adoption_allowed=False, cases=[],
                baseline="Actual Sampler.greedy_sample => torch.argmax, standalone provider; model binding not yet proven",
                source_sha256=hashlib.sha256((base.parent / "tianshu_greedy_argmax_20261014/implementation.py").read_bytes()).hexdigest(),
                argmax_cuda_registration=torch._C._dispatch_dump_table("aten::argmax"),
                flag_gems_source=str(Path(flag_gems.argmax.__code__.co_filename)),
                flag_gems_sha256=hashlib.sha256(Path(flag_gems.argmax.__code__.co_filename).read_bytes()).hexdigest())

    def save():
        (root / "result.json").write_text(json.dumps(data, indent=2) + "\n")

    save()
    for case_id, (batch, vocab, strided) in enumerate(((1,130560,False),(7,130560,False),(64,130560,False),(64,130560,True),(3,13,True))):
        for pattern in ("normal", "zero_tie", "nan_first", "infinity", "negative_infinity", "large"):
            torch.manual_seed(2026101400 + case_id)
            storage = torch.randn(batch, vocab + (29 if strided else 0), device="cuda", dtype=torch.float32)
            x = storage[:, :vocab]
            if pattern == "zero_tie":
                x.zero_(); x[:, 1] = 5.; x[:, -1] = 5.
            elif pattern == "nan_first":
                x[:, 2] = float("nan"); x[:, -1] = float("nan")
            elif pattern == "infinity":
                x[:, 3] = float("inf"); x[:, -1] = float("inf")
            elif pattern == "negative_infinity":
                x.fill_(-float("inf"))
            elif pattern == "large":
                x.mul_(1e30)
            before = x.view(torch.int32).clone()
            expected = x.cpu().argmax(-1)
            reference = Sampler.greedy_sample(x)
            assert reference.dtype == torch.int64 and torch.equal(reference.cpu(), expected)
            baseline_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(baseline_graph):
                baseline_output = Sampler.greedy_sample(x)
            baseline_graph.replay(); torch.cuda.synchronize()
            assert torch.equal(baseline_output.cpu(), expected)
            case = dict(batch=batch, vocab=vocab, strided=strided, pattern=pattern,
                        cpu_native_indices_equal=True, variants=[])
            for config in plans:
                entry = dict(configuration=config)
                graph = output = None
                try:
                    rng_before = torch.cuda.get_rng_state()
                    for _ in range(3):
                        output = (flag_gems.argmax(x, dim=-1) if config["provider"] == "explicit_flag_gems" else launch(x, dict(tile=1024 if batch == 1 else 2048,warps=4)))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = (flag_gems.argmax(x, dim=-1) if config["provider"] == "explicit_flag_gems" else launch(x, dict(tile=1024 if batch == 1 else 2048,warps=4)))
                    graph.replay(); torch.cuda.synchronize()
                    assert output.dtype == torch.int64 and output.shape == (batch,)
                    assert torch.equal(output.cpu(), expected)
                    assert torch.equal(before, x.view(torch.int32))
                    assert torch.equal(rng_before, torch.cuda.get_rng_state())
                    entry.update(cpu_native_and_graph_exact=True, input_bytes_unchanged=True, RNG_unchanged=True)
                    if pattern == "normal":
                        times = [[], []]
                        for repeat in range(7):
                            for index in ((0,1) if repeat % 2 == 0 else (1,0)):
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(50):
                                    (baseline_graph if index == 0 else graph).replay()
                                end.record(); end.synchronize()
                                times[index].append(start.elapsed_time(end) / 50)
                        b, c = map(statistics.median, times)
                        entry.update(baseline_ms=b, candidate_ms=c, time_change_percent=100*(c/b-1), raw_ms=times)
                except Exception as error:
                    entry["error"] = repr(error)
                case["variants"].append(entry)
            data["cases"].append(case);save()
            print(json.dumps(case), flush=True)
            del graph, baseline_graph, baseline_output, output, x, storage, before
            torch.cuda.empty_cache()
    data.update(complete=True, numerical_checks_passed=all(all(v.get("cpu_native_and_graph_exact",False) for v in c["variants"]) for c in data["cases"]))
    save()


if __name__ == "__main__":
    main()
