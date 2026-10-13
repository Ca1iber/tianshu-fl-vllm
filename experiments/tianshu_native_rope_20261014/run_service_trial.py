# SPDX-License-Identifier: Apache-2.0
"""Fresh 4K/1K trial: verified TopP foundation plus optional small RMS launch."""
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
    parser.add_argument("--numeric-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    base = Path(__file__).resolve().parents[1]
    trial = base / "tianshu_native_prefill_20261012/run_service_trial.py"
    topp_hook = base / "tianshu_topp_launch_20261013/hooks/sitecustomize.py"
    coherent = base / "tianshu_decode_tiles_coherent_20261013/hooks/sitecustomize.py"
    rms_hook = Path(__file__).with_name("hooks") / "sitecustomize.py"
    impl = Path(__file__).with_name("implementation.py")
    proof = json.loads((args.numeric_root / "result.json").read_text())
    smoke = json.loads((args.smoke_root / "result.json").read_text())
    assert smoke["phase"] == "complete" and smoke["binding_verified"] and smoke["gpu_idle_after"]
    assert smoke["source_sha256"] == proof["source_sha256"]
    assert proof["complete"] and proof["numerical_checks_passed"] and len(proof["cases"]) == 40
    assert hashlib.sha256(impl.read_bytes()).hexdigest() == proof["source_sha256"]
    assert impl.read_bytes() == (args.numeric_root / "implementation.py").read_bytes()
    assert rms_hook.read_bytes() == (args.numeric_root / "sitecustomize.py").read_bytes()
    hook = topp_hook if args.mode == "control" else rms_hook
    root.mkdir(parents=True)
    snapshot = root / "rope_source_snapshot"
    snapshot.mkdir()
    for name, path in [("run_service_trial.py", Path(__file__)), ("foundation_service_trial.py", trial),
                       ("active_sitecustomize.py", hook), ("topp_sitecustomize.py", topp_hook),
                       ("coherent_sitecustomize.py", coherent), ("sitecustomize.py", rms_hook),
                       ("implementation.py", impl)]:
        (snapshot / name).write_bytes(path.read_bytes())
    original = subprocess.Popen
    actual_env = dict(PYTHONPATH=str(hook.parent), VLLM_DISABLE_COMPILE_CACHE="1", TIANSHU_TOPP_LAUNCH_PROBE="1",
                      TIANSHU_TOPP_LAUNCH_RECORD_DIR=str(root / "topp_launch_records"),
                      TIANSHU_DECODE_COHERENT_PROBE="1",
                      TIANSHU_DECODE_COHERENT_RECORD_DIR=str(root / "decode_coherent_records"),
                      TIANSHU_NATIVE_ROPE_PROBE="1" if args.mode == "candidate" else "0",
                      TIANSHU_NATIVE_ROPE_RECORD_DIR=str(root / "rope_records"))

    def popen(command, *positional, **kwargs):
        if isinstance(command, list) and command[:2] == ["vllm", "serve"]:
            cfg_index = command.index("--compilation-config") + 1
            cfg = json.loads(command[cfg_index]); cfg["cache_dir"] = str(root / "compile_cache")
            command[cfg_index] = json.dumps(cfg)
            kwargs["env"] = dict(kwargs["env"], **actual_env)
        return original(command, *positional, **kwargs)

    spec = importlib.util.spec_from_file_location("_tianshu_rms_trial_foundation", trial)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.Popen = popen
    sys.argv = [str(trial), "--output", str(root), "--buckets", "1", "--gather", "1",
                "--bucket-trace", "2", "--hybrid", "1", "--adaptive", "1", "--tile", "1",
                "--native-prefill", "1", "--decode-launch", "1", "--max-batched-tokens", "2048",
                "--hist-count-bits", "off", "--address32", "0", "--prefill-stages", "1",
                "--input-tokens", "4096", "--output-tokens", "1024", "--concurrency", "64",
                "--requests", "64", "--rounds", "3", "--request-timeout", "3600"]
    try:
        code = module.main()
    finally:
        subprocess.Popen = original
    state = json.loads((root / "result.json").read_text())
    records = [json.loads(p.read_text()) for p in (root / "rope_records").glob("*.json")]
    topp = [json.loads(p.read_text()) for p in (root / "topp_launch_records").glob("route_*.json")]
    decode = [json.loads(p.read_text()) for p in (root / "decode_coherent_records").glob("producer_*.json")]
    reducers = [json.loads(p.read_text()) for p in (root / "decode_coherent_records").glob("reducer_*.json")]
    state.update(rope_mode=args.mode, actual_composed_environment=actual_env,
                 native_rope_records=records, topp_launch_records=topp,
                 decode_coherent_records=decode, decode_coherent_reducer_records=reducers,
                 rope_hook_sha256=hashlib.sha256(rms_hook.read_bytes()).hexdigest(),
                 rope_implementation_sha256=hashlib.sha256(impl.read_bytes()).hexdigest(),
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
        if args.mode == "candidate":
            assert records and any(r["query_row_stride"] == 2560 for r in records)
            assert all(r["source_sha256"] == proof["source_sha256"] for r in records)
        else:
            assert not records
    return code


if __name__ == "__main__":
    raise SystemExit(main())
