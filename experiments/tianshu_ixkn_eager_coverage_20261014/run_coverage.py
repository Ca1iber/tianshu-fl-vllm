# SPDX-License-Identifier: Apache-2.0
"""Direct-only profiler coverage negative/start controls; audit actual CSV afterward."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.output.resolve()
    assert not root.exists()
    hardware = subprocess.check_output(["ixsmi"], text=True)
    assert "No running processes found" in hardware
    root.mkdir(parents=True)
    (root / "hardware_before.txt").write_text(hardware)
    base = Path(__file__).resolve().parent
    for source in (Path(__file__), base / "probe_coverage.py"):
        (root / source.name).write_bytes(source.read_bytes())
    tool = Path("/usr/local/corex/bin/ixkn-cli")
    for flag in ("--help", "--version", "--list-sections", "--list-sets"):
        r = subprocess.run([str(tool), flag], capture_output=True, text=True, timeout=30)
        (root / (flag[2:] + ".txt")).write_text(r.stdout + r.stderr)
    state = dict(phase="running", runner_pid=os.getpid(), coverage_claimed=False,
                 started_utc=datetime.now(timezone.utc).isoformat(), stages={},
                 profiler_sha256=hashlib.sha256(tool.read_bytes()).hexdigest())
    child = None
    deadline = datetime.fromisoformat(json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 0

    def save():
        (root / "state.json").write_text(json.dumps(state, indent=2) + "\n")

    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise InterruptedError(str(signum))

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, stop)
    signal.alarm(min(900, remaining))
    save()
    try:
        for mode in ("no_start", "start"):
            dest = root / mode
            dest.mkdir()
            command = [str(tool), "--metrics", "time_duration", "--kernel-name", "tianshu_ixkn_coverage",
                       "--launch-count", "100", "--profile-from-start", "off", "--cache-control", "none",
                       "--csv", "--export-profile", str(dest / "trace"), sys.executable, "-u",
                       str(base / "probe_coverage.py"), "--output", str(dest / "workload")]
            if mode == "no_start":
                command.append("--no-start")
            state["current_stage"] = mode
            with (dest / "profiler.log").open("w") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state["child_pid"] = child.pid
                save()
                try:
                    code = child.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=15)
                    raise
            result = dest / "workload/result.json"
            stage = dict(command=command, exit_code=code, workload_result_produced=result.exists())
            if result.exists():
                stage["workload"] = json.loads(result.read_text())
            report = dest / "trace.ixkn-rep"
            if report.exists():
                imported = subprocess.run([str(tool), "--import-profile", str(report), "--csv"],
                                          capture_output=True, text=True, timeout=60)
                (dest / "import.csv").write_text(imported.stdout)
                (dest / "import.stderr").write_text(imported.stderr)
                stage["import_exit_code"] = imported.returncode
            state["stages"][mode] = stage
            save()
            assert result.exists() and stage["workload"]["numeric_exact"], mode
            if mode == "start":
                assert code == 0, "Profiler start-stage exited unsuccessfully"
                assert stage.get("import_exit_code") == 0, "Profiler report import failed"
            assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
        state["phase"] = "complete_pending_coverage_audit"
    except Exception as error:
        state.update(phase="failed", error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=15)
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
        (root / "hardware_after.txt").write_text(subprocess.check_output(["ixsmi"], text=True))
    return 0 if state["phase"] == "complete_pending_coverage_audit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
