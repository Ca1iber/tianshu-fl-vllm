# SPDX-License-Identifier: Apache-2.0
"""Fresh sampled 4K trial; both arms keep budget4096 and actual83 warm points."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "candidate"), required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    base = Path(__file__).resolve().parent
    foundation = base.parent / "tianshu_budget8192_buckets_20261014/run_service_trial.py"
    original_popen = subprocess.Popen
    actual_command = None
    actual_settings = None

    def popen(command, *positional, **kwargs):
        nonlocal actual_command, actual_settings
        if isinstance(command, list) and command[:2] == ["vllm", "serve"]:
            settings = dict(PYTHONPATH=str(base / "hooks"), TIANSHU_SWEEP_SERVICE_MODE=args.mode,
                            TIANSHU_SWEEP_RECORD_DIR=str(root / "sweep_records"))
            kwargs["env"] = dict(kwargs["env"], **settings)
            actual_command, actual_settings = list(command), settings
        return original_popen(command, *positional, **kwargs)

    spec = importlib.util.spec_from_file_location("_sweep_profile_trial", foundation)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_argv = sys.argv
    sys.argv = [str(foundation), "--output", str(root), "--mode", "control"]
    subprocess.Popen = popen
    try:
        code = module.main()
    finally:
        sys.argv = old_argv
        subprocess.Popen = original_popen
    state = json.loads((root / "result.json").read_text())
    state.update(sweep_mode=args.mode, inherited_budget_mode="control",
                 actual_service_environment=actual_settings, actual_service_command=actual_command,
                 active_hook=dict(path=str(base / "hooks/sitecustomize.py"),
                                  sha256=hashlib.sha256((base / "hooks/sitecustomize.py").read_bytes()).hexdigest()),
                 source_foundation_trial=str(foundation), source_foundation_sha256=hashlib.sha256(foundation.read_bytes()).hexdigest())
    if code == 0:
        assert state["phase"] == "complete"
        assert state["actual_budget"] == 4096 and state["actual_bucket_points"] == 83
        assert state["request_sampling_config"]["temperature"] == 1.0
        assert state["request_sampling_config"]["top_p"] == .95
        assert actual_command == state["command"]
        records = [json.loads(path.read_text()) for path in (root / "sweep_records").glob("route_*.json")]
        assert records and all(record["mode"] == args.mode for record in records)
        counts = [[0] * 256 for _ in range(3)]
        for record in records:
            raw = (root / "sweep_records" / ("counts_" + str(record["pid"]) + ".bin")).read_bytes()
            assert len(raw) == 3 * 256 * 8
            values = struct.unpack("<768Q", raw)
            for group in range(3):
                for batch in range(256):
                    counts[group][batch] += values[group * 256 + batch]
        assert sum(counts[0]) > 0 and sum(counts[1]) > 0
        assert all(counts[1][b] <= counts[0][b] for b in range(256))
        if args.mode == "control":
            assert sum(counts[2]) == 0
        else:
            assert sum(counts[2][1:8]) > 0 and sum(counts[2][8:]) == 0
        state.update(actual_sweep_records=records, actual_sweep_route_counts=counts,
                     count_scope="worker lifecycle including startup, canaries and benchmark",
                     sampler_local_sort_only=True, native_topp_path_unchanged=True,
                     actual_sweep_mode_verified=True)
    (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
