# SPDX-License-Identifier: Apache-2.0
"""Run the owned grid microprobe only after the existing reducer job really exits."""
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
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--upstream-pid", type=int, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    upstream = json.loads((args.upstream / "state.json").read_text())
    assert upstream["runner_pid"] == args.upstream_pid
    base = Path(__file__).resolve().parent
    sources = [Path(__file__), base / "probe_address.py", base / "address32.py",
               base.parent / "tianshu_decode_cache_policy_20261014/experimental_cache.py",
               base.parent / "tianshu_native_prefill_20261012/zero_initialization.py",
               base.parent / "tianshu_decode_low_warps_20261014/launch_router.py"]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    deadline = datetime.fromisoformat(json.loads(Path(
        "/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json"
    ).read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1800
    root.mkdir(parents=True)
    for source in sources:
        dest = root / "source_snapshot" / source.relative_to(base.parent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    state = dict(phase="waiting_for_reducer", runner_pid=os.getpid(), upstream_pid=args.upstream_pid,
                 upstream_root=str(args.upstream), stop_at_utc=deadline.isoformat(),
                 GPU_started=False, frozen_sources=frozen)
    child = None

    class Stopped(BaseException):
        pass

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temp = root / "state.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(root / "state.json")

    def stop(signum, frame):
        raise Stopped(str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.alarm(remaining)
    save()
    try:
        while True:
            current = json.loads((args.upstream / "state.json").read_text())
            assert current["runner_pid"] == args.upstream_pid
            proc = Path("/proc") / str(args.upstream_pid) / "stat"
            if not proc.exists() or proc.read_text().split()[2] == "Z":
                assert current["phase"] == "complete", current["phase"]
                assert Path("/tmp/tianshu_reducer_launch_20261014_manifest.json").exists()
                break
            time.sleep(8)
        assert (deadline - datetime.now(timezone.utc)).total_seconds() > 900
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in frozen.items())
        assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
        command = ["python3", "-u", str(base / "probe_address.py"), "--output", str(root / "micro"),
                   "--zero-init", "--b64-segments", "8"]
        with (root / "micro.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(phase="running_probe", child_pid=child.pid, command=command, GPU_started=True)
            save()
            assert child.wait() == 0
        result = json.loads((root / "micro/result.json").read_text())
        assert result["complete"] and result["numerical_checks_passed"]
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in frozen.items())
        assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
        state.update(phase="complete", complete=True, gpu_idle_after=True)
    except Stopped as error:
        state.update(phase="stopped", error=str(error))
    except Exception as error:
        state.update(phase="failed", error=repr(error))
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
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
        subprocess.run(["python3", str(base.parent / "archive_completed_pair.py"), "--single-probe",
                        "--source", str(root), "--label", "cv_address32_20261014"], check=True)
    return 0 if state["phase"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
