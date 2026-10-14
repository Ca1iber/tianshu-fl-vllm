# SPDX-License-Identifier: Apache-2.0
"""B64S8 whole-schema versus target-table SQL reflection, same4K load."""
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
    trial = base / "tianshu_greedy_screen_20261014/run_service_trial.py"
    topp_hook = base / "tianshu_decode_kv16_w2_20261014/hooks/sitecustomize.py"
    coherent = base / "tianshu_decode_tiles_coherent_20261013/hooks/sitecustomize.py"
    budget = 2048
    threshold = 0
    candidate_hook = Path(__file__).resolve().parent / "hooks/sitecustomize.py"
    hook = candidate_hook
    sql_enabled = args.mode == "candidate"
    root.mkdir(parents=True)
    snapshot = root / "budget_source_snapshot"
    snapshot.mkdir()
    for name,path in [("run_service_trial.py",Path(__file__)),("foundation_service_trial.py",trial),
                      ("topp_sitecustomize.py",topp_hook),("coherent_sitecustomize.py",coherent)]:
        (snapshot/name).write_bytes(path.read_bytes())
    for name in ("sitecustomize.py",):
        (snapshot / ("candidate_"+name)).write_bytes(candidate_hook.read_bytes())
    original = subprocess.Popen
    actual_env = dict(PYTHONPATH=str(hook.parent), TIANSHU_TOPP_LAUNCH_PROBE="1",
                      TIANSHU_TOPP_LAUNCH_RECORD_DIR=str(root / "topp_launch_records"),
                      TIANSHU_DECODE_COHERENT_PROBE="1",
                      TIANSHU_DECODE_COHERENT_RECORD_DIR=str(root / "decode_coherent_records"),
                      TIANSHU_IR_RMS_PROBE="0", TIANSHU_KV16_SERVICE_MODE="candidate",
                      TIANSHU_SQL_REFLECTION_PROBE="1" if sql_enabled else "0",
                      TIANSHU_SQL_REFLECTION_RECORD_DIR=str(root / "sql_reflection_records"),
                      TIANSHU_SEGMENT_POLICY_PROBE="1",
                      TIANSHU_SEGMENT_POLICY_RECORD_DIR=str(root / "segment_policy_records"))

    def popen(command, *positional, **kwargs):
        if isinstance(command, list) and command[:2] == ["vllm", "serve"]:
            command.extend(["--long-prefill-token-threshold", str(threshold)])
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
    policy = [json.loads(p.read_text()) for p in (root / "segment_policy_records").glob("choose_*.json")]
    state.update(sql_reflection_enabled=sql_enabled, sql_mode=args.mode, segment_mode="candidate", segment_policy_records=policy, lowwarps_mode=args.mode, actual_budget=budget, actual_threshold=threshold, actual_composed_environment=actual_env,
                 topp_launch_records=topp,
                 decode_coherent_records=decode, decode_coherent_reducer_records=reducers,
                 foundation_topp_hook_sha256=hashlib.sha256(topp_hook.read_bytes()).hexdigest(),
                 foundation_coherent_hook_sha256=hashlib.sha256(coherent.read_bytes()).hexdigest(),
                 active_hook=dict(path=str(hook), sha256=hashlib.sha256(hook.read_bytes()).hexdigest()),
                 new_independent_math_not_run=True)
    (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if code == 0:
        assert all(r["configuration"] == dict(num_warps=16, num_stages=1) for r in topp)
        assert decode and reducers
        for r in decode:
            expected_tile = 16 if r["batch"] in (8,16,32,40,48,56,64) else 32
            expected_warps = 1 if expected_tile == 16 else 2 if 8 <= r["batch"] <= 64 else 4
            assert r["configuration"] == dict(TILE_SIZE=expected_tile, num_warps=expected_warps, num_stages=1)
            assert r["compiled_metadata"]["num_warps"] == expected_warps
            expected_segments = 64 if r["batch"] == 1 else 4 if 16 <= r["batch"] <= 64 else 16
            expected_segments = 8 if r["batch"] == 64 else expected_segments
            assert r["segments"] == expected_segments
        assert policy
        assert any(r["batch"]==64 and r["selected_segments"]==8 for r in policy)
        assert all(r["selected_segments"]==(8 if r["batch"]==64 else r["baseline_segments"]) for r in policy)
        assert any(r["batch"]==64 and r["segments"]==8 for r in decode)
        assert {(r["batch"], r["segments"]) for r in decode} == {(r["batch"], r["segments"]) for r in reducers}
        assert all(r["actual_tile"] == (16 if r["batch"] in (8,16,32,40,48,56,64) else 32) for r in reducers)
        assert state["request_sampling_config"] == dict(temperature=0.,top_p=1.,repetition_penalty=1.,logprobs=None,ignore_eos=True,stream=True)
        assert state["max_batched_tokens"] == budget
        assert state["command"][state["command"].index("--max-num-batched-tokens")+1] == str(budget)
        configured = [json.loads(p.read_text()) for p in (root / "bucket_trace_records").glob("configured_*.json")]
        warmed = [json.loads(p.read_text()) for p in (root / "bucket_trace_records").glob("warmup_completed_*.json")]
        expected = 75
        assert any(r["enabled"] and max(r["points"]) == budget and len(r["points"]) == expected for r in configured)
        assert any(r["complete"] and len(r["calls"]) == expected for r in warmed)
        state["actual_bucket_points"] = expected
        (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if code == 0:
        records=[json.loads(p.read_text()) for p in (root / "sql_reflection_records").glob("route_*.json")]
        if sql_enabled:
            assert records and all(r["reflection_scope"]=="one_requested_table" and not r["configuration_values_modified"] for r in records)
            assert all(r["helper_sha256"]=="4cfc6a34e29e0cc220b83d79898c1da4d986e35d4eb0796c37947b1631064c02" for r in records)
        else:assert not records
        state.update(actual_sql_reflection_records=records,actual_sql_mode_verified=True,
                     workflow_cold_time_trial=True,official_benchmark=False)
        (root / "result.json").write_text(json.dumps(state,indent=2)+"\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
