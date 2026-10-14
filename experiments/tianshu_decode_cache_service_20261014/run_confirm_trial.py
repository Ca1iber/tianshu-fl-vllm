# SPDX-License-Identifier: Apache-2.0
"""Compare native Decode cache policy with .cv at the adopted4096/83 profile."""
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
    parser.add_argument("--mode", choices=("control", "candidate"), required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    base = Path(__file__).resolve().parent
    foundation = base / "foundation_confirm_trial.py"
    original_popen = subprocess.Popen
    actual_command = None
    actual_settings = None

    def popen(command, *positional, **kwargs):
        nonlocal actual_command, actual_settings
        if isinstance(command, list) and command[:2] == ["vllm", "serve"]:
            settings = dict(PYTHONPATH=str(base / "hooks"), TIANSHU_CACHE_SERVICE_MODE=args.mode)
            kwargs["env"] = dict(kwargs["env"], **settings)
            actual_command, actual_settings = list(command), settings
        return original_popen(command, *positional, **kwargs)

    spec = importlib.util.spec_from_file_location("_cache_profile_trial", foundation)
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
    state.update(cache_mode=args.mode, inherited_budget_mode="control",
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
        producers = state["decode_coherent_records"]
        assert producers and all(record["cache_service_mode"] == args.mode for record in producers)
        expected_sha = ("4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e" if args.mode == "control"
                        else "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289")
        assert all(record["generated_source_sha256"] == expected_sha for record in producers)
        assert all(record["actual_cache_modifier"] == (None if args.mode == "control" else ".cv") for record in producers)
        state.update(actual_cache_mode_verified=True, cache_modifier=None if args.mode == "control" else ".cv",
                     actual_decode_generated_sha256=expected_sha, Decode_KV_load_policy_only=True,
                     sampling_native_topp_and_prefill_unchanged=True)
    (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
