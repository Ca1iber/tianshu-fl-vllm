# SPDX-License-Identifier: Apache-2.0
"""Run max-sequence A/B only after .cv quality and formal validation complete."""
import argparse
import ast
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
    for name in ("output", "quality-chain", "math-root", "official-root", "trial-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists() and not args.trial_root.exists()
    base = Path(__file__).resolve().parent
    upstream = json.loads((args.quality_chain / "state.json").read_text())
    upstream_pid = upstream["runner_pid"]
    paths = ast.literal_eval(next(node.value for node in ast.walk(ast.parse((base / "run_service_pair.py").read_text()))
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "relatives" for target in node.targets)))
    sources = [Path(__file__), *(base.parent / relative for relative in paths)]
    frozen = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    assert len(frozen) == 17
    deadline = datetime.fromisoformat(json.loads(Path(
        "/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json"
    ).read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1800
    root.mkdir(parents=True)
    for path in sources:
        dest = root / "source_snapshot" / path.resolve().relative_to(base.parent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    state = dict(phase="waiting_for_quality", runner_pid=os.getpid(), upstream_pid=upstream_pid,
        frozen_sources=frozen, stop_at_utc=deadline.isoformat(), GPU_started=False,
        upstream_root=str(args.quality_chain), trial_root=str(args.trial_root))
    child = None

    class StopRequested(BaseException): pass
    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temp = root / "state.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(root / "state.json")
    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None and child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        raise StopRequested(str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM): signal.signal(sig, stop)
    signal.alarm(remaining)
    save()
    try:
        while True:
            current = json.loads((args.quality_chain / "state.json").read_text())
            assert current["runner_pid"] == upstream_pid
            proc = Path("/proc") / str(upstream_pid) / "stat"
            alive = proc.exists() and proc.read_text().split()[2] != "Z"
            if not alive:
                if current["phase"] != "complete":
                    state.update(phase="upstream_not_validated", upstream_phase=current["phase"])
                break
            time.sleep(8)
        if state["phase"] == "waiting_for_quality":
            formal = json.loads((args.official_root / "comparison.json").read_text())
            if not formal["adopt_as_best"]:
                state["phase"] = "upstream_not_adoptable"
            elif (deadline - datetime.now(timezone.utc)).total_seconds() < 1200:
                state["phase"] = "insufficient_remaining_lease"
            else:
                assert Path("/tmp/tianshu_decode_cache_cv_official4k_20261014_manifest.json").exists()
                assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in frozen.items())
                assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
                command = ["python3", "-u", str(base / "run_service_pair.py"), "--output", str(args.trial_root),
                           "--quality-root", str(args.math_root), "--official-root", str(args.official_root)]
                with (root / "service_pair.log").open("w") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state.update(phase="running_service", child_pid=child.pid, GPU_started=True)
                    save()
                    assert child.wait() == 0
                state["phase"] = "complete"
    except StopRequested as error:
        state.update(phase="stopped", error=str(error))
    except Exception as error:
        state.update(phase="failed", error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        save()
        (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    return 1 if state["phase"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
