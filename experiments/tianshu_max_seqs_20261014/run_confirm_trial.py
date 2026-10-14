# SPDX-License-Identifier: Apache-2.0
"""Compare automatic max_num_seqs with64 on the same independently validated.cv profile."""
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
            if args.mode == "candidate": command.extend(["--max-num-seqs", "64"])
            settings = dict(PYTHONPATH=str(base / "hooks"), TIANSHU_CACHE_SERVICE_MODE="candidate",
                            TIANSHU_MAX_SEQS_MODE=args.mode,
                            TIANSHU_MAX_SEQS_RECORD_DIR=str(root / "max_seqs_records"))
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
    state.update(max_seqs_mode=args.mode, cache_mode="candidate", inherited_budget_mode="control",
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
        assert producers and all(record["cache_service_mode"] == "candidate" for record in producers)
        expected_sha = "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
        assert all(record["generated_source_sha256"] == expected_sha for record in producers)
        assert all(record["actual_cache_modifier"] == ".cv" for record in producers)
        state.update(actual_cache_mode_verified=True, cache_modifier=".cv",
                     actual_decode_generated_sha256=expected_sha, max_num_seqs_only=True,
                     sampling_native_topp_and_prefill_unchanged=True)
    if code == 0:
        records = [json.loads(path.read_text()) for path in (root / "max_seqs_records").glob("worker_*.json")]
        assert records and all(record["mode"] == args.mode for record in records)
        sizes = {record["max_num_seqs"] for record in records}
        assert len(sizes) == 1 and min(sizes) >= 64
        if args.mode == "candidate": assert sizes == {64}
        assert all(record["max_num_batched_tokens"] == 4096 and record["model_max_len"] == 131072 for record in records)
        assert all(record["worker_source_sha256"] == state["installed_sources"]["worker.py"]["sha256"] for record in records)
        state.update(actual_worker_config_records=records, actual_max_num_seqs=sizes.pop(), actual_max_num_seqs_verified=True)
    (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
