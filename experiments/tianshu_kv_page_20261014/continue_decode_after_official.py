# SPDX-License-Identifier: Apache-2.0
"""Run frozen page16/page32 Decode probe after the B64S8 formal pipeline has exited."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--upstream-pid", type=int, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    lease = json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())
    deadline = datetime.fromisoformat(lease["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1800
    root.mkdir(parents=True)
    base = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), base/"tianshu_kv_page_20261014/probe_decode_pages.py",
             base/"tianshu_decode_low_warps_20261014/launch_router.py",
             base/"tianshu_native_prefill_20261012/zero_initialization.py"]
    snapshots = root / "source_snapshot"
    snapshots.mkdir()
    frozen = {}
    for path in paths:
        raw = path.read_bytes()
        frozen[str(path)] = hashlib.sha256(raw).hexdigest()
        dest = snapshots/path.relative_to(base)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(raw)
    state = dict(phase="waiting_for_formal_completion", runner_pid=os.getpid(),
                 upstream_root=str(args.wait_for), upstream_pid=args.upstream_pid,
                 stop_at_utc=deadline.isoformat(), frozen_sources=frozen, jobs=[],
                 adopted=False, official_benchmark=False)
    child = None

    class StopRequested(BaseException):
        pass

    def save():
        p = root / "state.tmp"
        p.write_text(json.dumps(state, indent=2) + "\n")
        p.replace(root / "state.json")

    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise StopRequested(str(signum))

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, stop)
    signal.alarm(remaining)
    save()
    try:
        while True:
            upstream = json.loads((args.wait_for / "state.json").read_text())
            assert upstream["runner_pid"] == args.upstream_pid
            status = upstream["phase"]
            if status in ("failed", "stopped", "quality_rejected"):
                raise StopRequested("Upstream stopped or failed; followups not started")
            if status == "complete":
                proc = Path("/proc") / str(args.upstream_pid) / "stat"
                alive = proc.exists() and proc.read_text().split()[2] != "Z"
                if not alive and "No running processes found" in subprocess.check_output(["ixsmi"], text=True):
                    break
            time.sleep(5)
        formal = Path(upstream["official_root"])
        official = json.loads((formal/"comparison.json").read_text())
        assert official["complete"] and official["official_benchmark"]
        assert Path("/tmp/tianshu_kv16_segments_B64S8_official4k_20261014_manifest.json").exists()
        state["upstream_completion_verified"] = True
        state["formal_total_tok_s"] = official["total_tok_s"]
        try:
            connection = socket.create_connection(("127.0.0.1", 9031), timeout=2)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError("Port9031 remains occupied")
        b64_segments="8" if official["adopt_as_best"] else "4"
        state["b64_segments_from_formal_comparison"]=int(b64_segments)
        state["baseline_tag_not_automatically_changed"]=True
        jobs = [("kv_page32_decode",paths[1],1800,["--zero-init","--b64-segments",b64_segments],"kv_page32_decode_after_B64S8_20261014")]
        for name, script, limit, extra, label in jobs:
            assert (deadline - datetime.now(timezone.utc)).total_seconds() > limit
            assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in frozen.items())
            state.update(phase="running", current_job=name)
            with (root / (name + ".log")).open("w") as log:
                env = dict(os.environ)
                env.pop("PYTHONPATH", None)
                child = subprocess.Popen(["python3", "-u", str(script), "--output", str(root / name)] + extra,
                                         stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                state["child_pid"] = child.pid
                save()
                code = child.wait(timeout=limit)
            state["jobs"].append(dict(name=name, exit_code=code))
            save()
            assert code == 0, name
            result = json.loads((root / name / "result.json").read_text())
            assert result["complete"] is True
            state["jobs"][-1]["result"] = {
                k: result.get(k) for k in ("complete", "numerical_checks_passed", "valid_page_sizes", "scope", "error")
            }
            hardware_after=subprocess.check_output(["ixsmi"],text=True)
            assert "No running processes found" in hardware_after
            (root/name/"hardware_after.txt").write_text(hardware_after)
            subprocess.run(["python3", "/workspace/vllm-plugin-FL/experiments/archive_completed_pair.py",
                            "--single-probe", "--source", str(root / name),
                            "--label", label], check=True,
                           stdout=subprocess.DEVNULL)
        state["phase"] = "complete"
    except StopRequested as error:
        state.update(phase="stopped", error=str(error))
    except Exception as error:
        state.update(phase="failed", error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        save()
        (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    return 0 if state["phase"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
