# SPDX-License-Identifier: Apache-2.0
"""Candidate-only capture feasibility and three-round screen; reused reference is explicit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-control", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    reference = json.loads((args.reference_control / "result.json").read_text())
    assert reference["phase"] == "complete" and len(reference["short_4k_rounds"]) == 3
    assert reference["actual_graph_mode"] == "FULL_DECODE_ONLY" and reference["common_gpu_memory_utilization"] == .70
    base = Path(__file__).resolve().parent
    probe = json.loads(Path("/workspace/benchmark-results/20261014_weak_alias_probe_tianshu/result.json").read_text())
    assert probe["complete"] and probe["CPU_owner_lifetime_verified"] and probe["changed_input_graphs_bitwise_verified"]
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    deadline = datetime.fromisoformat(json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1200
    root.mkdir(parents=True)
    shutil.copytree(args.reference_control, root / "reference_control")
    relatives = json.loads((base / "frozen_relatives.json").read_text()) + ["tianshu_piecewise_sparse_20261014/frozen_relatives.json"]
    frozen = {}
    for relative in relatives:
        path = base.parent / relative
        data = path.read_bytes()
        frozen[str(path)] = hashlib.sha256(data).hexdigest()
        destination = root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    state = dict(phase="preparing", runner_pid=os.getpid(), frozen_sources=frozen,
                 official_benchmark=False, adopted=False, fresh_paired_control=False,
                 reference_reused=True, preliminary=True, stop_at_utc=deadline.isoformat())
    child = None
    class StopRequested(BaseException): pass
    def stop(signum, frame):
        if child is not None and child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        raise StopRequested(str(signum))
    def save():
        temp = root / "state.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(root / "state.json")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM): signal.signal(sig, stop)
    signal.alarm(remaining)
    try:
        with (root / "candidate_runner.log").open("w") as log:
            child = subprocess.Popen(["python3", "-u", str(base / "run_service_trial.py"), "--mode", "candidate",
                                      "--output", str(root / "candidate")], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(phase="running_candidate", child_pid=child.pid)
            save()
            assert child.wait() == 0
        candidate = json.loads((root / "candidate/result.json").read_text())
        assert candidate["phase"] == "complete" and candidate["actual_graph_mode_verified"]
        alias = [json.loads(p.read_text()) for p in (root / "candidate/piecewise_records").glob("alias_*.json")]
        assert alias and all(r["actual_shared_pointer"] and r["mode"] == "candidate" for r in alias)
        assert all(r["extension"]["cpp_source_sha256"] == probe["extension"]["cpp_source_sha256"] for r in alias)
        assert candidate["installed_sources"] == reference["installed_sources"]
        assert candidate["request_sampling_config"] == reference["request_sampling_config"]
        values = [[r["total_tokens_per_second"] for r in x["short_4k_rounds"]] for x in (reference, candidate)]
        scores = [statistics.mean(x[1:]) for x in values]
        proof = dict(complete=True, official_benchmark=False, adopted=False, preliminary=True,
                     fresh_paired_control=False, reference_reused=True, excluded_rounds=[1],
                     reference_rounds=values[0], candidate_rounds=values[1], reference_total_tok_s=scores[0],
                     candidate_total_tok_s=scores[1], gain_percent=100*(scores[1]/scores[0]-1),
                     candidate_successful_requests=192, reference_control=str(args.reference_control),
                     requires_fresh_independent_confirmation_before_adoption=True)
        (root / "comparison.json").write_text(json.dumps(proof, indent=2) + "\n")
        state.update(phase="complete", gain_percent=proof["gain_percent"])
    except (Exception, StopRequested) as error:
        state.update(phase="failed", error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
        save()
        (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["phase"] == "complete":
        subprocess.run(["python3", str(base.parent / "archive_completed_pair.py"), "--single-probe", "--source", str(root),
                        "--label", "piecewise_sparse_candidate_20261014"], check=True)
    return 0 if state["phase"] == "complete" else 1


if __name__ == "__main__": raise SystemExit(main())
