# SPDX-License-Identifier: Apache-2.0
"""Fresh4K service A/B: default async scheduling versus disabled async on currentCV profile."""
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
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    quality = json.loads((args.quality_root / "accuracy_audit.json").read_text())
    formal = json.loads((args.official_root / "comparison.json").read_text())
    assert quality["complete"] and quality["total"] == 105 and quality["accuracy"] > .95
    assert quality["fresh_math_for_decode_cache_cv"] and formal["fresh_math_for_decode_cache_cv"]
    assert formal["adopt_as_best"] and formal["requests_successful"] == 1024
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    deadline = datetime.fromisoformat(json.loads(Path(
        "/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json"
    ).read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1800
    root.mkdir(parents=True)
    base = Path(__file__).resolve().parent
    relatives = [
        "tianshu_async_schedule_20261016/run_service_pair.py",
        "tianshu_async_schedule_20261016/run_service_trial.py",
        "tianshu_async_schedule_20261016/hooks/sitecustomize.py",
        "tianshu_decode_cache_service_20261014/hooks/sitecustomize.py",
        "tianshu_decode_cache_service_20261014/hooks/decode_sitecustomize.py",
        "tianshu_decode_cache_policy_20261014/experimental_cache.py",
        "tianshu_budget8192_buckets_20261014/run_service_trial.py",
        "tianshu_budget8192_buckets_20261014/foundation_service_trial.py",
        "tianshu_budget8192_buckets_20261014/hooks/sitecustomize.py",
        "tianshu_budget8192_buckets_20261014/extend_buckets.py",
        "tianshu_decode_kv16_w2_20261014/hooks/sitecustomize.py",
        "tianshu_decode_kv16_w2_20261014/hooks/decode_sitecustomize.py",
        "tianshu_decode_tiles_coherent_20261013/hooks/sitecustomize.py",
        "tianshu_kv16_segments_20261014/hooks/sitecustomize.py",
        "tianshu_sql_workflow_20261014/hooks/sitecustomize.py",
        "tianshu_sql_workflow_20261014/reflect_one.py",
    ]
    relatives += ["tianshu_max_seqs_20261014/hooks/sitecustomize.py"]
    assert len(relatives) == 17
    frozen = {}
    for relative in relatives:
        path = base.parent / relative
        raw = path.read_bytes()
        frozen[str(path)] = hashlib.sha256(raw).hexdigest()
        dest = root / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    state = dict(phase="running", runner_pid=os.getpid(), official_benchmark=False, adopted=False,
        workload=dict(input_tokens=4096, output_tokens=1024, concurrency=64, requests_per_round=64),
        rounds=3, excluded_rounds=[1], frozen_sources=frozen, stop_at_utc=deadline.isoformat())
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
        p = root / "state.tmp"
        p.write_text(json.dumps(state, indent=2) + "\n")
        p.replace(root / "state.json")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, stop)
    signal.alarm(remaining)
    save()
    try:
        trials = []
        for mode in ("control", "candidate"):
            assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in frozen.items())
            state["current_job"] = mode
            with (root / (mode + "_runner.log")).open("w") as log:
                child = subprocess.Popen(["python3", "-u", str(base / "run_service_trial.py"),
                    "--mode", mode, "--output", str(root / mode)], stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                state["child_pid"] = child.pid
                save()
                assert child.wait() == 0, mode
            trial = json.loads((root / mode / "result.json").read_text())
            assert trial["phase"] == "complete" and trial["async_mode"] == mode
            assert trial["actual_cache_mode_verified"] and trial["actual_max_num_seqs_verified"] and trial["actual_async_config_verified"]
            assert trial["actual_budget"] == 4096 and trial["actual_bucket_points"] == 83
            assert trial["request_sampling_config"]["temperature"] == 1.0
            assert trial["request_sampling_config"]["top_p"] == .95
            assert len(trial["short_4k_rounds"]) == 3
            for row in trial["short_4k_rounds"]:
                assert row["successful_requests"] == len(row["responses"]) == 64
                assert all(r["usage"]["prompt_tokens"] == 4096 and r["usage"]["completion_tokens"] == 1024 for r in row["responses"])
            trials.append(trial)
        aa, bb = trials
        for key in ("workload", "installed_sources", "request_sampling_config"):
            assert aa[key] == bb[key], key
        command_control, command_candidate = list(aa["command"]), list(bb["command"])
        assert "--no-async-scheduling" not in command_control
        assert command_candidate.count("--no-async-scheduling") == 1
        command_candidate.remove("--no-async-scheduling")
        assert command_control == command_candidate
        assert aa["actual_max_num_seqs"] == bb["actual_max_num_seqs"] == 256
        assert aa["actual_async_scheduling"] is True and bb["actual_async_scheduling"] is False
        values = [[row["total_tokens_per_second"] for row in trial["short_4k_rounds"]] for trial in trials]
        scores = [statistics.mean(x[1:]) for x in values]
        comparison = dict(complete=True, official_benchmark=False, adopted=False, control_reused=False,
            excluded_rounds=[1], control_rounds=values[0], candidate_rounds=values[1],
            control_total_tok_s=scores[0], candidate_total_tok_s=scores[1],
            gain_percent=100 * (scores[1] / scores[0] - 1), successful_requests=384,
            control_budget=4096, candidate_budget=4096, warm_points=[83,83],
            request_sampling_config=aa["request_sampling_config"], frozen_sources=frozen,
            actual_commands_differ_only_async_flag=True, async_scheduling_only=True,
            control_max_num_seqs=aa["actual_max_num_seqs"], candidate_max_num_seqs=bb["actual_max_num_seqs"],
            matched_quality_root=str(args.quality_root), matched_official_root=str(args.official_root),
            sampling_native_topp_and_prefill_unchanged=True, new_independent_math_not_run=True,
            control_decode_generated_sha256=aa["actual_decode_generated_sha256"],
            candidate_decode_generated_sha256=bb["actual_decode_generated_sha256"])
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
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        save()
        (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["phase"] == "complete":
        assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
        subprocess.run(["python3", str(base.parent / "archive_completed_pair.py"), "--single-probe",
            "--source", str(root), "--label", "async_schedule_service_20261016"], check=True)
    return 0 if state["phase"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
