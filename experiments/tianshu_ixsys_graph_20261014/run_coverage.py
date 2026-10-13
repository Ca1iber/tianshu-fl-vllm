# SPDX-License-Identifier: Apache-2.0
"""Collect IXsys trace for known Graph kernel counts; coverage is audited separately."""
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
    lease = json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())
    remaining = int((datetime.fromisoformat(lease["stop_at_utc"]) - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 600
    root.mkdir(parents=True)
    base = Path(__file__).resolve().parent
    for p in (Path(__file__), base / "probe_coverage.py"):
        (root / p.name).write_bytes(p.read_bytes())
    command = ["/usr/local/corex/bin/ixsys", "-t", "cuda", "-o", str(root / "trace"),
               "--profile-from-start", "off", "--print-gpu-summary", "--print-gpu-trace",
               "python3", "-u", str(base / "probe_coverage.py"), "--output", str(root / "workload")]
    state = dict(phase="preparing", runner_pid=os.getpid(), official_benchmark=False, adopted=False,
                 command=command, profiler_coverage_verified=False,
                 stop_at_utc=lease["stop_at_utc"], source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (Path(__file__), base / "probe_coverage.py")})
    child = None

    class StopRequested(BaseException):
        pass

    def save():
        p = root / "state.tmp"; p.write_text(json.dumps(state, indent=2) + "\n"); p.replace(root / "state.json")

    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise StopRequested(str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.alarm(min(remaining, 600)); save()
    try:
        (root / "ixsys_help.txt").write_text(subprocess.check_output(["/usr/local/corex/bin/ixsys", "--help"], text=True))
        (root / "ixsys_version.txt").write_text(subprocess.check_output(["/usr/local/corex/bin/ixsys", "--version"], text=True))
        (root / "hardware_before.txt").write_text(subprocess.check_output(["ixsmi"], text=True))
        env = dict(os.environ); env.pop("PYTHONPATH", None)
        with (root / "ixsys.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            state.update(phase="profiling", child_pid=child.pid); save()
            code = child.wait(timeout=540)
        state["ixsys_exit_code"] = code
        assert code == 0
        workload = json.loads((root / "workload/result.json").read_text())
        assert workload["complete"] and workload["graph_output_correct"]
        state.update(phase="complete", complete=True, workload=workload)
    except (Exception, StopRequested) as error:
        state.update(phase="failed", error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
        state["gpu_idle_after"] = "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
        save(); (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    return 0 if state["phase"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
