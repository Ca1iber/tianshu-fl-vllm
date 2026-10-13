# SPDX-License-Identifier: Apache-2.0
"""Fresh TopP-best control/candidate 4K small pair, preserving seeds and usage."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
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
    deadline = datetime.fromisoformat(json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 3600
    root.mkdir(parents=True)
    base = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve(), base / "run_service_trial.py", 
             base.parent / "tianshu_topp_launch_20261013/hooks/sitecustomize.py",
             base.parent / "tianshu_decode_tiles_coherent_20261013/hooks/sitecustomize.py",
             base.parent / "tianshu_native_prefill_20261012/run_service_trial.py"]
    frozen = {}
    for path in paths:
        raw = path.read_bytes(); frozen[str(path)] = hashlib.sha256(raw).hexdigest()
        dest = root / "source_snapshot" / path.relative_to(base.parent)
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
    state = dict(phase="running", runner_pid=os.getpid(), official_benchmark=False, adopted=False,
                 workload=dict(input_tokens=4096, output_tokens=1024, concurrency=64, requests_per_round=64),
                 rounds=3, excluded_rounds=[1], stop_at_utc=deadline.isoformat(), frozen_sources=frozen)
    child = None

    class StopRequested(BaseException):
        pass

    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise StopRequested(str(signum))

    def save():
        p = root / "state.tmp"; p.write_text(json.dumps(state, indent=2) + "\n"); p.replace(root / "state.json")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, stop)
    signal.alarm(remaining); save()
    try:
        for mode in ("control", "candidate"):
            assert datetime.now(timezone.utc) < deadline
            assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in frozen.items())
            state["current_job"] = mode
            command = ["python3", "-u", str(base / "run_service_trial.py"), "--mode", mode,
                       "--output", str(root / mode)]
            with (root / (mode + "_runner.log")).open("w") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state["child_pid"] = child.pid; save()
                assert child.wait() == 0, mode
        ctrl = json.loads((root / "control/result.json").read_text())
        cand = json.loads((root / "candidate/result.json").read_text())
        assert ctrl["phase"] == cand["phase"] == "complete"
        assert ctrl["fairness_mode"] == "control" and cand["fairness_mode"] == "candidate"
        for key in ("workload", "installed_sources", "foundation_topp_hook_sha256", "foundation_coherent_hook_sha256"):
            assert ctrl[key] == cand[key], key
        assert ctrl["actual_budget"] == cand["actual_budget"] == 2048
        assert ctrl["actual_threshold"] == 0 and cand["actual_threshold"] == 1024
        assert ctrl["actual_bucket_points"] == cand["actual_bucket_points"] == 75
        commands = [list(t["command"]) for t in (ctrl,cand)]
        for cmd in commands:
            idx = cmd.index("--long-prefill-token-threshold") + 1
            cmd[idx] = "THRESHOLD"
        assert commands[0] == commands[1]
        assert len(ctrl["short_4k_rounds"]) == len(cand["short_4k_rounds"]) == 3
        for aa, bb in zip(ctrl["short_4k_rounds"], cand["short_4k_rounds"]):
            assert aa["successful_requests"] == bb["successful_requests"] == 64
            assert [r["seed"] for r in aa["responses"]] == [r["seed"] for r in bb["responses"]]
            assert all(r["usage"]["prompt_tokens"] == 4096 and r["usage"]["completion_tokens"] == 1024 for row in (aa, bb) for r in row["responses"])
        values = [[r["total_tokens_per_second"] for r in d["short_4k_rounds"]] for d in (ctrl, cand)]
        scores = [statistics.mean(v[1:]) for v in values]
        comparison = dict(complete=True, official_benchmark=False, adopted=False, control_reused=False,
                          excluded_rounds=[1], control_rounds=values[0], candidate_rounds=values[1],
                          control_total_tok_s=scores[0], candidate_total_tok_s=scores[1],
                          gain_percent=100 * (scores[1] / scores[0] - 1), successful_requests=384,
                          greedy_cases_equal=sum(a["text"] == b["text"] for a, b in zip(ctrl["rare_cases"], cand["rare_cases"])),
                          global_budget=2048, control_threshold=0, candidate_threshold=1024, bucket_points=[75,75],
                          foundation_topp_hook_sha256=cand["foundation_topp_hook_sha256"],
                          foundation_coherent_hook_sha256=cand["foundation_coherent_hook_sha256"],
                          new_independent_math_not_run=True, frozen_sources=frozen)
        (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        state.update(phase="complete", gain_percent=comparison["gain_percent"])
    except StopRequested as error:
        state.update(phase="stopped", error=str(error))
    except Exception as error:
        state.update(phase="failed", error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
        save(); (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["phase"] == "complete":
        assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
        subprocess.run(["python3", str(base.parent / "archive_completed_pair.py"), "--single-probe",
                        "--source", str(root), "--label", "prefill_fairness_service_20261014"], check=True)
    return 0 if state["phase"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
