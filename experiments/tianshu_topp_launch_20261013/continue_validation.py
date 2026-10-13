# SPDX-License-Identifier: Apache-2.0
"""After confirmed TopP service gain, rerun independent105 and original4K only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    deadline = datetime.fromisoformat(json.loads(Path(
        "/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json"
    ).read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 7200
    signal.alarm(remaining)
    root.mkdir(parents=True)
    base = Path(__file__).resolve().parent
    foundation = base.parent / "tianshu_decode_tiles_coherent_20261013"
    paths = [Path(__file__), base / "run_math_eval.py", base / "run_official_4k.py", base / "hooks/sitecustomize.py",
             foundation / "run_math_eval.py", foundation / "audit_math_eval.py", foundation / "run_official_4k.py",
             foundation / "audit_official.py", foundation / "hooks/sitecustomize.py"]
    frozen = {}
    for p in paths:
        raw = p.read_bytes()
        frozen[str(p)] = hashlib.sha256(raw).hexdigest()
        dest = root / "validation_source_snapshot" / p.relative_to(base.parent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    state = dict(phase="waiting_for_confirm5", runner_pid=os.getpid(), jobs=[], adopted=False,
                 service_root=str(args.service_root), stop_at_utc=deadline.isoformat(), frozen_sources=frozen,
                 formal_workloads=[4096], no_old_accuracy_reuse=True)
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

    def run(name, command):
        nonlocal child
        assert datetime.now(timezone.utc) < deadline
        assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
        for p, sha in frozen.items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha, p
        state.update(phase="running", current_job=name)
        with (root / f"{name}_runner.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["child_pid"] = child.pid
            save()
            code = child.wait()
        state["jobs"].append(dict(name=name, exit_code=code))
        save()
        assert code == 0, name
        assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)

    save()
    try:
        while not (args.service_root / "result.json").exists():
            assert datetime.now(timezone.utc) < deadline
            time.sleep(5)
        assert json.loads((args.service_root / "result.json").read_text())["phase"] == "complete"
        c = json.loads((args.service_root / "comparison.json").read_text())
        clear = c["gain_percent"] >= 2 and min(c["candidate_rounds"][2:]) > max(c["control_rounds"][2:])
        state.update(service_gain_percent=c["gain_percent"], confirmed_service_gain_clear=clear)
        save()
        if clear:
            run("math", ["python3", "-u", str(base / "run_math_eval.py"), "--output", str(root / "math"),
                         "--service-root", str(args.service_root)])
            q = json.loads((root / "math/accuracy_audit.json").read_text())
            assert q["fresh_math_for_topp_launch"]
            state.update(math_correct=q["correct"], math_total=105, math_accuracy=q["accuracy"],
                         quality_passed=q["official_requirement_passed"])
            save()
            if q["official_requirement_passed"]:
                assert (deadline - datetime.now(timezone.utc)).total_seconds() > 3600
                run("official4096", ["python3", "-u", str(base / "run_official_4k.py"), "--output", str(root / "official4096"),
                    "--service-root", str(args.service_root), "--math-results", str(root / "math")])
            else:
                state["formal_skipped_reason"] = "NewindependentTopP accuracy didnotexceed95%"
        else:
            state["formal_skipped_reason"] = "Confirm5 last3 service gain not clearly above2% with nonoverlap"
        state["phase"] = "complete"
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
    return 0 if state["phase"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
