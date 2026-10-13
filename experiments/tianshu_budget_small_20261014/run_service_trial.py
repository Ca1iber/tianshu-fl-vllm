# SPDX-License-Identifier: Apache-2.0
"""Current-best 4K/1K trial with budget2048 or budget1024 and bounded warmup."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["control", "candidate"], required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    base = Path(__file__).resolve().parents[1]
    trial = Path(__file__).with_name("local_service_trial.py")
    topp_hook = base / "tianshu_topp_launch_20261013/hooks/sitecustomize.py"
    coherent = base / "tianshu_decode_tiles_coherent_20261013/hooks/sitecustomize.py"
    budget = 2048 if args.mode == "control" else 1024
    hook = topp_hook
    root.mkdir(parents=True)
    snapshot = root / "budget_source_snapshot"
    snapshot.mkdir()
    for name,path in [("run_service_trial.py",Path(__file__)),("local_service_trial.py",trial),
                      ("topp_sitecustomize.py",topp_hook),("coherent_sitecustomize.py",coherent)]:
        (snapshot/name).write_bytes(path.read_bytes())
    original = subprocess.Popen
    actual_env = dict(PYTHONPATH=str(hook.parent), TIANSHU_TOPP_LAUNCH_PROBE="1",
                      TIANSHU_TOPP_LAUNCH_RECORD_DIR=str(root / "topp_launch_records"),
                      TIANSHU_DECODE_COHERENT_PROBE="1",
                      TIANSHU_DECODE_COHERENT_RECORD_DIR=str(root / "decode_coherent_records"),
                      TIANSHU_IR_RMS_PROBE="0")

    def popen(command, *positional, **kwargs):
        if isinstance(command, list) and command[:2] == ["vllm", "serve"]:
            kwargs["env"] = dict(kwargs["env"], **actual_env)
        return original(command, *positional, **kwargs)

    spec = importlib.util.spec_from_file_location("_tianshu_budget_trial_foundation", trial)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.Popen = popen
    sys.argv = [str(trial), "--output", str(root), "--buckets", "1", "--gather", "1",
                "--bucket-trace", "2", "--hybrid", "1", "--adaptive", "1", "--tile", "1",
                "--native-prefill", "1", "--decode-launch", "1", "--max-batched-tokens", str(budget),
                "--hist-count-bits", "off", "--address32", "0", "--prefill-stages", "1",
                "--input-tokens", "4096", "--output-tokens", "1024", "--concurrency", "64",
                "--requests", "64", "--rounds", "3", "--request-timeout", "3600"]
    try:
        code = module.main()
    finally:
        subprocess.Popen = original
    state = json.loads((root / "result.json").read_text())
    topp = [json.loads(p.read_text()) for p in (root / "topp_launch_records").glob("route_*.json")]
    decode = [json.loads(p.read_text()) for p in (root / "decode_coherent_records").glob("producer_*.json")]
    reducers = [json.loads(p.read_text()) for p in (root / "decode_coherent_records").glob("reducer_*.json")]
    state.update(budget_mode=args.mode, actual_budget=budget, actual_composed_environment=actual_env,
                 topp_launch_records=topp,
                 decode_coherent_records=decode, decode_coherent_reducer_records=reducers,
                 foundation_topp_hook_sha256=hashlib.sha256(topp_hook.read_bytes()).hexdigest(),
                 foundation_coherent_hook_sha256=hashlib.sha256(coherent.read_bytes()).hexdigest(),
                 active_hook=dict(path=str(hook), sha256=hashlib.sha256(hook.read_bytes()).hexdigest()),
                 new_independent_math_not_run=True)
    (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if code == 0:
        assert topp and all(r["configuration"] == dict(num_warps=16, num_stages=1) for r in topp)
        assert decode and reducers and all(r["configuration"] == dict(TILE_SIZE=32, num_warps=4, num_stages=1) for r in decode)
        assert {(r["batch"], r["segments"]) for r in decode} == {(r["batch"], r["segments"]) for r in reducers}
        assert all(r["actual_tile"] == 32 for r in reducers)
        assert state["max_batched_tokens"] == budget
        assert state["command"][state["command"].index("--max-num-batched-tokens")+1] == str(budget)
        configured = [json.loads(p.read_text()) for p in (root / "bucket_trace_records").glob("configured_*.json")]
        warmed = [json.loads(p.read_text()) for p in (root / "bucket_trace_records").glob("warmup_completed_*.json")]
        expected = module.expected_bucket_count(budget)
        assert any(r["enabled"] and max(r["points"]) == budget and len(r["points"]) == expected for r in configured)
        assert any(r["complete"] and len(r["calls"]) == expected for r in warmed)
        state["actual_bucket_points"] = expected
        (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
