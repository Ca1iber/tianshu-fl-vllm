# SPDX-License-Identifier: Apache-2.0
"""Fresh4K service comparison for checked coherent L0 DecodeKV32."""
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
    parser.add_argument("--micro-root",type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists()
    proof=json.loads((args.micro_root/'result.json').read_text());assert proof['complete'] and len(proof['cases'])==7
    for c in proof['cases']:
        v=next(v for v in c['variants'] if v['label']=='w16s1')
        assert v.get('exact_equal') or v.get('cpu_semantic_checks_passed'), 'Changed masks require independentCPU validation'
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    deadline = datetime.fromisoformat(json.loads(Path(
        "/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json"
    ).read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining >= 3600
    signal.alarm(remaining)
    root.mkdir(parents=True)
    state = dict(phase="running", runner_pid=os.getpid(), official_benchmark=False, adopted=False,
                 input_tokens=4096, output_tokens=1024, concurrency=64, requests_per_round=64,
                 rounds=3, excluded_rounds=[1], stop_at_utc=deadline.isoformat())
    child = None

    class StopRequested(BaseException):
        pass

    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise StopRequested(f"Signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, stop)

    def save():
        p = root / "state.tmp"
        p.write_text(json.dumps(state, indent=2) + "\n")
        p.replace(root / "state.json")

    base = Path(__file__).resolve().parents[1]
    trial = base / "tianshu_native_prefill_20261012/run_service_trial.py"
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for p in (Path(__file__), Path(__file__).with_name("run_service_trial.py"), trial,
              Path(__file__).with_name("hooks") / "sitecustomize.py"):
        (snapshot / ("original_service_trial.py" if p == trial else p.name)).write_bytes(p.read_bytes())
    save()
    try:
        commands=[(name,["python3","-u",str(Path(__file__).with_name("run_service_trial.py")),
                          "--output",str(root/name),"--mode",name]) for name in ["control","candidate"]]
        for name, command in commands:
            assert datetime.now(timezone.utc) < deadline
            state["current_job"] = name
            with (root / f"{name}_runner.log").open("w") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state["child_pid"] = child.pid
                save()
                assert child.wait() == 0, name
            assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
        control = json.loads((root / "control/result.json").read_text())
        candidate = json.loads((root / "candidate/result.json").read_text())
        assert control["phase"] == candidate["phase"] == "complete"
        for key in ("workload", "installed_sources", "decode_hook", "tile_hook", "bucket_trace_hook"):
            assert control[key] == candidate[key], key
        assert control['coherent_hook_sha256']==candidate['coherent_hook_sha256']
        assert control['topp_mode']=='control' and candidate['topp_mode']=='candidate'
        for value in (control, candidate):
            assert value["address32"] == "0" and value["prefill_stages"] == 1
            assert value["watermark"] == value["decode_graph_max"] == 0
            assert value["histogram_count_bits"] == "off"
        assert len(control["rare_cases"]) == len(candidate["rare_cases"]) == 7
        greedy_equal = sum(a["text"] == b["text"] for a, b in zip(control["rare_cases"], candidate["rare_cases"]))
        assert len(control["short_4k_rounds"]) == len(candidate["short_4k_rounds"]) == 3
        for left, right in zip(control["short_4k_rounds"], candidate["short_4k_rounds"]):
            assert left["successful_requests"] == right["successful_requests"] == 64
            assert [r["seed"] for r in left["responses"]] == [r["seed"] for r in right["responses"]]
            assert all(r["usage"]["prompt_tokens"] == 4096 and r["usage"]["completion_tokens"] == 1024
                       for group in (left, right) for r in group["responses"])
        scores = [statistics.mean(r["total_tokens_per_second"] for r in d["short_4k_rounds"][1:])
                  for d in (control, candidate)]
        summary = dict(complete=True, official_benchmark=False, adopted=False, excluded_rounds=[1],
                       control_reused=False, control_total_tok_s=scores[0], candidate_total_tok_s=scores[1],
                       gain_percent=100 * (scores[1] / scores[0] - 1),
                       control_rounds=[r["total_tokens_per_second"] for r in control["short_4k_rounds"]],
                       candidate_rounds=[r["total_tokens_per_second"] for r in candidate["short_4k_rounds"]],
                       successful_requests=384, greedy_cases_equal=greedy_equal,
                       actual_decode_coherent_records=candidate["decode_coherent_records"],
                       hook_sha256=candidate["decode_coherent_hook_sha256"], independent_math_not_run=True,
                       source_micro_sha256=proof['native_source_sha256'],
                       topp_records=candidate['topp_launch_records'],
                       candidate_topp_hook_sha256=candidate['topp_hook_sha256'],
                       actual_reducer_records=candidate['decode_coherent_reducer_records'])
        assert candidate["topp_hook_sha256"] == hashlib.sha256((snapshot / "sitecustomize.py").read_bytes()).hexdigest()
        (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
        state.update(phase="complete", comparison=summary)
    except StopRequested as error:
        state.update(phase="stopped", error=str(error))
    except Exception as error:
        state.update(phase="failed", error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        save()
        (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["phase"] == "complete":
        subprocess.run(["python3", str(base / "archive_completed_pair.py"), "--single-probe",
                        "--source", str(root), "--label", "topp_launch_service_20261013"], check=True)
    return 0 if state["phase"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
