# SPDX-License-Identifier: Apache-2.0
"""Check CoreX external timing events in replayed CUDA Graph before profiling."""
import argparse
from datetime import datetime, timezone
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
    deadline = datetime.fromisoformat(json.loads(Path(
        "/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json"
    ).read_text())["stop_at_utc"])
    assert (deadline - datetime.now(timezone.utc)).total_seconds() > 300
    signal.alarm(300)
    root.mkdir(parents=True)
    (root / "probe_external_events.py").write_bytes(Path(__file__).read_bytes())
    data = dict(complete=False, supported=False, official_benchmark=False, adopted=False,
                scope="Backend capability check, not model performance")
    import torch
    try:
        with torch.inference_mode():
            torch.manual_seed(2026101388)
            x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            y0 = torch.empty_like(x)
            z0 = torch.empty_like(x)
            y1 = torch.empty_like(x)
            z1 = torch.empty_like(x)
            for _ in range(3):
                torch.mm(x, w, out=y0)
                torch.add(y0, 1, out=z0)
            baseline = torch.cuda.CUDAGraph()
            with torch.cuda.graph(baseline):
                torch.mm(x, w, out=y0)
                torch.add(y0, 1, out=z0)
            start = torch.cuda.Event(enable_timing=True, external=True)
            middle = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            instrumented = torch.cuda.CUDAGraph()
            with torch.cuda.graph(instrumented):
                start.record()
                torch.mm(x, w, out=y1)
                middle.record()
                torch.add(y1, 1, out=z1)
                end.record()
            # Record a deliberately different interval outside the graph, then
            # prove replay updates these same event handles rather than using stale timestamps.
            start.record()
            for _ in range(50):
                torch.mm(x, w, out=y1)
            middle.record()
            torch.add(y1, 1, out=z1)
            end.record()
            end.synchronize()
            outside_mm_ms = start.elapsed_time(middle)
            instrumented.replay()
            torch.cuda.synchronize()
            replay_mm_ms = start.elapsed_time(middle)
            assert 0 < replay_mm_ms < outside_mm_ms / 5, "Graph did not refresh external event timestamps"
            times = dict(baseline=[], instrumented=[], segment_mm=[], segment_add=[], segment_all=[])
            for i in range(9):
                order = [("baseline", baseline), ("instrumented", instrumented)]
                if i % 2:
                    order.reverse()
                for name, graph in order:
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(50):
                        graph.replay()
                    b.record()
                    b.synchronize()
                    times[name].append(a.elapsed_time(b) / 50)
                    if name == "instrumented":
                        assert start.query() and middle.query() and end.query()
                        times["segment_mm"].append(start.elapsed_time(middle))
                        times["segment_add"].append(middle.elapsed_time(end))
                        times["segment_all"].append(start.elapsed_time(end))
                assert torch.equal(z0, z1), "Events changed output"
            assert all(t > 0 for key, vals in times.items() for t in vals), "Replay timestamps were not populated"
            assert all(abs(a + b - c) < .002 for a, b, c in zip(
                times["segment_mm"], times["segment_add"], times["segment_all"]))
            medians = {k: statistics.median(v) for k, v in times.items()}
            data.update(complete=True, supported=True, replay_timestamps_positive=True, bitwise_output_equal=True,
                        outside_event_mm_ms=outside_mm_ms, overwritten_replay_mm_ms=replay_mm_ms,
                        stale_timestamp_check_passed=True,
                        times_ms=times, medians_ms=medians,
                        overhead_percent=100 * (medians["instrumented"] / medians["baseline"] - 1))
    except Exception as error:
        data.update(complete=True, supported=False, error=repr(error))
    (root / "result.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
