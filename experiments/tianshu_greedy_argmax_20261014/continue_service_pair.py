# SPDX-License-Identifier: Apache-2.0
"""Fresh greedy0 matched current-best control/candidate with actual request routes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-from", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    deadline = datetime.fromisoformat(json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 3600
    root.mkdir(parents=True)
    base = Path(__file__).resolve().parent
    original_control = args.control_from.resolve()
    old_root = original_control.parent
    old_state = json.loads((old_root / "result.json").read_text())
    assert old_state["phase"] == "failed" and old_state["current_job"] == "control"
    control = json.loads((original_control / "result.json").read_text())
    assert control["phase"] == "complete" and len(control["short_4k_rounds"]) == 3
    assert all(x["successful_requests"] == 64 and len(x["responses"]) == 64 for x in control["short_4k_rounds"])
    assert all(z["usage"] == dict(prompt_tokens=4096, completion_tokens=1024, total_tokens=5120)
               for x in control["short_4k_rounds"] for z in x["responses"])
    assert control["request_sampling_config"] == dict(temperature=0., top_p=1., repetition_penalty=1., logprobs=None, ignore_eos=True, stream=True)
    assert not control["topp_launch_records"]
    assert any(x["phase"] == "request" and x["batch"] == 64 and not x["selected"] for x in control["actual_argmax_records"])
    warmed = [json.loads(p.read_text()) for p in (original_control / "bucket_trace_records").glob("warmup_completed_*.json")]
    assert any(x["complete"] and len(x["calls"]) == 75 for x in warmed)
    old_trial = old_root / "source_snapshot/tianshu_greedy_argmax_20261014/run_service_trial.py"
    old_text = old_trial.read_text()
    old_assert = 'assert topp and all(r["configuration"] == dict(num_warps=16, num_stages=1) for r in topp)'
    new_assert = 'assert all(r["configuration"] == dict(num_warps=16, num_stages=1) for r in topp)'
    assert old_text.count(old_assert) == 1
    assert old_text.replace(old_assert, new_assert) == Path(__file__).with_name("run_service_trial.py").read_text()
    for path, digest in old_state["frozen_sources"].items():
        rel = Path(path).relative_to("/workspace/vllm-plugin-FL/experiments")
        saved = old_root / "source_snapshot" / rel
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
        if path.endswith("tianshu_greedy_argmax_20261014/run_service_trial.py"):
            continue
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    shutil.copytree(original_control, root / "control")
    shutil.copytree(old_root / "source_snapshot", root / "prior_control_failure/source_snapshot")
    for name in ("result.json", "state.json", "control_runner.log", "failure_audit.json"):
        shutil.copy2(old_root / name, root / "prior_control_failure" / name)
    original_result = (original_control / "result.json").read_bytes()
    (root / "control/original_result_before_postvalidation_recheck.json").write_bytes(original_result)
    for source in original_control.rglob("*"):
        if source.is_file():
            copied = root / "control" / source.relative_to(original_control)
            assert copied.read_bytes() == source.read_bytes()
    control["actual_bucket_points"] = 75
    control["postvalidation_rechecked_after_topP_guard_fix"] = True
    (root / "control/result.json").write_text(json.dumps(control, indent=2) + "\n")
    recovery = dict(control_reused_from=str(original_control), original_control_result_sha256=hashlib.sha256(original_result).hexdigest(),
                    original_trial_sha256=hashlib.sha256(old_trial.read_bytes()).hexdigest(),
                    current_trial_sha256=hashlib.sha256(Path(__file__).with_name("run_service_trial.py").read_bytes()).hexdigest(),
                    only_trial_change_is_post_inference_topP_nonempty_assertion=True,
                    original_failure_phase_preserved=True, all_existing_control_raw_bytes_copied_identically=True,
                    successful_control_requests_verified=192, actual75warm_and_greedyB64_verified=True,
                    control_gpu_requests_not_rerun=True)
    (root / "control_recovery_audit.json").write_text(json.dumps(recovery, indent=2) + "\n")
    paths = [Path(__file__).resolve(), base / "run_service_trial.py", base / "implementation.py",
             base / "hooks/sitecustomize.py",
             base.parent / "tianshu_greedy_screen_20261014/run_service_trial.py",
             base.parent / "tianshu_decode_low_warps_20261014/hooks/sitecustomize.py",
             base.parent / "tianshu_decode_low_warps_20261014/hooks/decode_sitecustomize.py"]
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
        for mode in ("candidate",):
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
        assert ctrl["argmax_mode"] == "control" and cand["argmax_mode"] == "candidate"
        for key in ("workload", "installed_sources", "foundation_topp_hook_sha256", "foundation_coherent_hook_sha256", "request_sampling_config"):
            assert ctrl[key] == cand[key], key
        assert ctrl["actual_budget"] == cand["actual_budget"] == 2048
        assert ctrl["actual_threshold"] == cand["actual_threshold"] == 0
        assert ctrl["actual_bucket_points"] == cand["actual_bucket_points"] == 75
        assert ctrl["command"] == cand["command"]
        assert len(ctrl["short_4k_rounds"]) == len(cand["short_4k_rounds"]) == 3
        for aa, bb in zip(ctrl["short_4k_rounds"], cand["short_4k_rounds"]):
            assert aa["successful_requests"] == bb["successful_requests"] == 64
            assert [r["seed"] for r in aa["responses"]] == [r["seed"] for r in bb["responses"]]
            assert all(r["usage"]["prompt_tokens"] == 4096 and r["usage"]["completion_tokens"] == 1024 for row in (aa, bb) for r in row["responses"])
        values = [[r["total_tokens_per_second"] for r in d["short_4k_rounds"]] for d in (ctrl, cand)]
        scores = [statistics.mean(v[1:]) for v in values]
        comparison = dict(complete=True, official_benchmark=False, adopted=False, control_reused=True, control_recovery=recovery,
                          excluded_rounds=[1], control_rounds=values[0], candidate_rounds=values[1],
                          control_total_tok_s=scores[0], candidate_total_tok_s=scores[1],
                          gain_percent=100 * (scores[1] / scores[0] - 1), successful_requests=384,
                          greedy_cases_equal=sum(a["text"] == b["text"] for a, b in zip(ctrl["rare_cases"], cand["rare_cases"])),
                          request_sampling_config=cand["request_sampling_config"], global_budget=2048, control_threshold=0, candidate_threshold=0, candidate_decode_warps=2, bucket_points=[75,75],
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
                        "--source", str(root), "--label", "greedy_argmax_service_recovered_20261014"], check=True)
    return 0 if state["phase"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
