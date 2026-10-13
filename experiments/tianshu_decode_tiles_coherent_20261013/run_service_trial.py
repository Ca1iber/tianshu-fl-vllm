# SPDX-License-Identifier: Apache-2.0
"""Wrap a4K candidate with checked L0 Decode32 and its matching reducer."""
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
    hook = Path(__file__).with_name("hooks") / "sitecustomize.py"
    snapshot = root / "decode_coherent_source_snapshot"
    snapshot.mkdir()
    for p in (Path(__file__), trial, hook):
        (snapshot / ("original_service_trial.py" if p == trial else p.name)).write_bytes(p.read_bytes())
    original = subprocess.Popen

    def popen(command, *positional, **kwargs):
        if isinstance(command, list) and command[:2] == ["vllm", "serve"]:
            env = dict(kwargs["env"])
            env.update(PYTHONPATH=str(hook.parent), TIANSHU_DECODE_COHERENT_PROBE="1",
                       TIANSHU_DECODE_COHERENT_RECORD_DIR=str(root / "decode_coherent_records"))
            kwargs["env"] = env
        return original(command, *positional, **kwargs)

    spec = importlib.util.spec_from_file_location("_tianshu_decode_coherent_trial", trial)
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
    records = [json.loads(p.read_text()) for p in (root / "decode_coherent_records").glob("producer_*.json")]
    reducers = [json.loads(p.read_text()) for p in (root / "decode_coherent_records").glob("reducer_*.json")]
    state.update(decode_coherent_hook_sha256=hashlib.sha256(hook.read_bytes()).hexdigest(),
                 decode_coherent_records=records, decode_coherent_reducer_records=reducers, decode_coherent_only=True)
    (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if code == 0:
        assert records and reducers, "Coherent Decode route was not exercised"
        assert all(r["configuration"] == dict(TILE_SIZE=32, num_warps=4, num_stages=1) for r in records)
        assert all(r["actual_tile"] == 32 for r in reducers)
        assert {(r["batch"], r["segments"]) for r in records} == {(r["batch"], r["segments"]) for r in reducers}
    return code


if __name__ == "__main__":
    raise SystemExit(main())
