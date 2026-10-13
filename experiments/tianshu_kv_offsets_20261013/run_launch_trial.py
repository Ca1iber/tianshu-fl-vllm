# SPDX-License-Identifier: Apache-2.0
"""Wrap the verified 4K trial without modifying its requests or arithmetic."""
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
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    base = Path(__file__).resolve().parents[1]
    trial = base / "tianshu_native_prefill_20261012/run_service_trial.py"
    hook = Path(__file__).with_name("hooks_launch") / "sitecustomize.py"
    snapshot = root / "kv_launch_source_snapshot"
    snapshot.mkdir()
    for p in (Path(__file__), trial, hook):
        (snapshot / p.name).write_bytes(p.read_bytes())
    original = subprocess.Popen

    def popen(command, *positional, **kwargs):
        if isinstance(command, list) and command[:2] == ["vllm", "serve"]:
            env = dict(kwargs["env"])
            env.update(PYTHONPATH=str(hook.parent), TIANSHU_KV_LAUNCH_PROBE="1",
                       TIANSHU_KV_LAUNCH_RECORD_DIR=str(root / "kv_launch_records"))
            kwargs["env"] = env
        return original(command, *positional, **kwargs)

    spec = importlib.util.spec_from_file_location("_tianshu_kv_launch_trial", trial)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.Popen = popen
    sys.argv = [str(trial), "--output", str(root), "--buckets", "1", "--gather", "1",
                "--bucket-trace", "2", "--hybrid", "1", "--adaptive", "1", "--tile", "1",
                "--native-prefill", "1", "--decode-launch", "1", "--max-batched-tokens", "2048",
                "--hist-count-bits", "off", "--address32", "0", "--prefill-stages", "1",
                "--input-tokens", "4096", "--output-tokens", "1024", "--concurrency", "64",
                "--requests", "64", "--rounds", "5", "--request-timeout", "3600"]
    try:
        code = module.main()
    finally:
        subprocess.Popen = original
    state = json.loads((root / "result.json").read_text())
    records = [json.loads(p.read_text()) for p in (root / "kv_launch_records").glob("route_*.json")]
    state.update(kv_launch_hook_sha256=hashlib.sha256(hook.read_bytes()).hexdigest(),
                 kv_launch_records=records, kv_launch_only=True)
    (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if code == 0:
        assert records and any(r["tokens"] == 2048 for r in records), "KV launch route was not exercised"
        assert all(r["configuration"] == dict(num_warps=4, num_stages=1) for r in records)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
