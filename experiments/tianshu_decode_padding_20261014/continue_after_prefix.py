# SPDX-License-Identifier: Apache-2.0
"""Run Decode-padding pair after prefix-cache screen, unless prefix needs confirmation."""
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
    paths = [Path(__file__).resolve(),base/"tianshu_decode_padding_20261014/run_service_pair.py",
             base/"tianshu_decode_padding_20261014/run_service_trial.py",base/"tianshu_decode_padding_20261014/hooks/sitecustomize.py",
             base/"tianshu_decode_kv16_w2_20261014/hooks/sitecustomize.py",
             base/"tianshu_decode_kv16_w2_20261014/hooks/decode_sitecustomize.py",
             base/"tianshu_kv16_segments_20261014/hooks/sitecustomize.py",
             base/"tianshu_native_prefill_20261012/hooks/sitecustomize.py",
             base/"tianshu_native_prefill_20261012/zero_initialization.py",
             base/"tianshu_fa2_tiles_20261012/hooks/sitecustomize.py",
             base/"tianshu_buckets75_recheck_20261012/joint_hooks/sitecustomize.py",
             base/"tianshu_buckets75_recheck_20261012/hooks/sitecustomize.py",
             base/"tianshu_decode_low_warps_20261014/launch_router.py"]
    snapshots = root / "source_snapshot"
    snapshots.mkdir()
    frozen = {}
    for path in paths:
        raw = path.read_bytes()
        frozen[str(path)] = hashlib.sha256(raw).hexdigest()
        dest = snapshots/path.relative_to(base)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(raw)
    state = dict(phase="waiting_for_prefix_cache_completion", runner_pid=os.getpid(),
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
            if status in ("failed", "stopped"):
                raise StopRequested("Upstream stopped or failed; followups not started")
            if status == "complete":
                proc = Path("/proc") / str(args.upstream_pid) / "stat"
                alive = proc.exists() and proc.read_text().split()[2] != "Z"
                if not alive and "No running processes found" in subprocess.check_output(["ixsmi"], text=True):
                    break
            time.sleep(5)
        comparison=json.loads((args.wait_for/"prefix_cache_service/comparison.json").read_text())
        assert comparison["complete"] and comparison["prefix_cache_enabled"]==[True,False]
        assert comparison["actual_engine_config_verified"] and comparison["successful_requests"]==384
        assert Path("/tmp/tianshu_prefix_cache_S8_service_20261014_manifest.json").exists()
        state["upstream_completion_verified"]=True
        state["prefix_cache_screen_gain_percent"]=comparison["gain_percent"]
        if comparison["gain_percent"]>=2:
            state.update(phase="held_for_prefix_confirmation",reason="Promising prefix-cache screen needs independent5+5, no padding job started")
            save()
            return 0
        try:
            connection = socket.create_connection(("127.0.0.1", 9031), timeout=2)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError("Port9031 remains occupied")
        jobs=[("decode_padding_service",paths[1],3600,[],"decode_padding_S8_service_20261014")]
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
            assert result["phase"] == "complete"
            state["jobs"][-1]["result"] = {
                k: result.get(k) for k in ("phase", "gain_percent", "error")
            }
            hardware_after=subprocess.check_output(["ixsmi"],text=True)
            assert "No running processes found" in hardware_after
            (root/name/"hardware_after.txt").write_text(hardware_after)
            # The service-pair runner creates its own immutable archive before exit.
            assert Path("/tmp/tianshu_decode_padding_S8_service_20261014_manifest.json").exists()
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
