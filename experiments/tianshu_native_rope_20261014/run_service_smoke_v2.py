# SPDX-License-Identifier: Apache-2.0
"""Verify real-model IR provider selection before spending time on benchmarks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve(); assert not root.exists()
    assert json.loads((args.numeric_root / "result.json").read_text())["numerical_checks_passed"]
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    proof = json.loads((args.numeric_root / "result.json").read_text())
    assert proof["complete"] and proof["numerical_checks_passed"] and len(proof["cases"]) == 40
    base = Path(__file__).resolve().parent
    assert hashlib.sha256((base / "implementation.py").read_bytes()).hexdigest() == proof["source_sha256"]
    assert (base / "implementation.py").read_bytes() == (args.numeric_root / "implementation.py").read_bytes()
    assert (base / "hooks/sitecustomize.py").read_bytes() == (args.numeric_root / "sitecustomize.py").read_bytes()
    deadline = datetime.fromisoformat(json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds()); assert remaining > 1800
    root.mkdir(parents=True)
    snapshot = root / "source_snapshot"; snapshot.mkdir()
    for p in [Path(__file__), base / "implementation.py", base / "hooks/sitecustomize.py"]:
        (snapshot / p.name).write_bytes(p.read_bytes())
    provider = "tianshu_native_rope"
    command = ["vllm", "serve", "/workspace/MiniCPM5-2B", "--port", "9031", "--compilation-config",
               '{"cudagraph_mode":"FULL_DECODE_ONLY"}', "--served-model-name", "minicpm",
               "--gpu-memory-utilization", "0.85", "--max-model-len", "131072",
               "--max-num-batched-tokens", "2048"]
    cfg_index = command.index("--compilation-config") + 1
    cfg = json.loads(command[cfg_index]); cfg["cache_dir"] = str(root / "compile_cache")
    command[cfg_index] = json.dumps(cfg)
    env = dict(os.environ, VLLM_PLUGINS="fl", PYTHONUNBUFFERED="1", VLLM_DISABLE_COMPILE_CACHE="1",
               PYTHONPATH=str(base / "hooks"), TIANSHU_NATIVE_ROPE_PROBE="1", TIANSHU_NATIVE_ROPE_RECORD_DIR=str(root / "rope_records"),
               VLLM_FL_ILUVATAR_HYBRID_ATTENTION="1", FLAGGEMS_ILUVATAR_COMPACT_VARLEN="1",
               VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS="1", VLLM_FL_USE_FLAGGEMS_ATTN="0",
               VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN="0", VLLM_FL_ILUVATAR_PREFILL_BUCKETS="1",
               VLLM_FL_ILUVATAR_LOGITS_GATHER="1", TIANSHU_TOPP_LAUNCH_PROBE="1",
               TIANSHU_TOPP_LAUNCH_RECORD_DIR=str(root / "topp_records"), TIANSHU_DECODE_COHERENT_PROBE="1",
               TIANSHU_DECODE_COHERENT_RECORD_DIR=str(root / "decode_records"), TIANSHU_NATIVE_PREFILL_PROBE="1",
               TIANSHU_NATIVE_PREFILL_RECORD_DIR=str(root / "prefill_records"), TIANSHU_DECODE_LAUNCH_PROBE="1",
               TIANSHU_DECODE_LAUNCH_RECORD_DIR=str(root / "fallback_decode_records"), TIANSHU_FA2_TILE_PROBE="1",
               TIANSHU_FA2_TILE_RECORD_DIR=str(root / "tile_records"), TIANSHU_BUCKET_RECHECK_TRACE="1",
               TIANSHU_BUCKET_TRACE_DIR=str(root / "bucket_records"), TRITON_PRINT_AUTOTUNING="1")
    for key in ("TIANSHU_ADDRESS32_PROBE", "TIANSHU_PREFILL_STAGE2_PROBE", "TIANSHU_RMS_LAUNCH_PROBE", "TIANSHU_IR_RMS_PROBE"):
        env[key] = "0"
    state = dict(phase="preparing", runner_pid=os.getpid(), official_benchmark=False, adopted=False,
                 command=command, source_sha256=proof["source_sha256"], compile_cache_disabled=True)
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
    signal.alarm(min(remaining, 1800)); save()
    try:
        try:
            c = socket.create_connection(("127.0.0.1", 9031), timeout=2)
        except OSError:
            pass
        else:
            c.close(); raise RuntimeError("Port9031 occupied")
        with (root / "server.log").open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(phase="starting", server_pid=child.pid); save()
        while True:
            assert child.poll() is None, "Server exited"
            try:
                with urllib.request.urlopen("http://127.0.0.1:9031/v1/models", timeout=3) as response:
                    models = json.load(response)
                if any(x["id"] == "minicpm" for x in models["data"]):
                    break
            except Exception:
                pass
            time.sleep(3)
        state["phase"] = "requesting"; save()
        rng = random.Random(2026101493)
        payload = dict(model="minicpm", prompt=[rng.randrange(512, 120000) for _ in range(4096)],
                       max_tokens=16, temperature=0, ignore_eos=True)
        request = urllib.request.Request("http://127.0.0.1:9031/v1/completions", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        (root / "response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        assert result["usage"]["prompt_tokens"] == 4096 and result["usage"]["completion_tokens"] == 16
        records = [json.loads(p.read_text()) for p in (root / "rope_records").glob("*.json")]
        assert records and any(r["query_row_stride"] == 2560 for r in records)
        assert all(r["source_sha256"] == proof["source_sha256"] for r in records)
        state.update(phase="complete", complete=True, actual_rope_records=records, usage=result["usage"],
                     binding_verified=True, new_independent_math_not_run=True)
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
