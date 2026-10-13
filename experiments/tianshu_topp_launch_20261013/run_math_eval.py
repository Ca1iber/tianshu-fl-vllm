# SPDX-License-Identifier: Apache-2.0
"""Independently rerun105 for TopP launch on verified coherent Decode foundation."""
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
    parser.add_argument("--service-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    comparison = json.loads((args.service_root / "comparison.json").read_text())
    service = json.loads((args.service_root / "candidate/result.json").read_text())
    assert comparison["complete"] and comparison["gain_percent"] >= 2 and service["phase"] == "complete"
    base = Path(__file__).resolve().parents[1]
    trial = base / "tianshu_decode_tiles_coherent_20261013/run_math_eval.py"
    hook = Path(__file__).with_name("hooks") / "sitecustomize.py"
    coherent = base / "tianshu_decode_tiles_coherent_20261013/hooks/sitecustomize.py"
    assert hashlib.sha256(hook.read_bytes()).hexdigest() == service["topp_hook_sha256"]
    assert hashlib.sha256(coherent.read_bytes()).hexdigest() == service["coherent_hook_sha256"]
    root.mkdir(parents=True)
    snapshot = root / "topp_source_snapshot"
    snapshot.mkdir()
    for name, p in [("run_math_eval.py", Path(__file__)), ("foundation_math_runner.py", trial),
                    ("topp_sitecustomize.py", hook), ("coherent_sitecustomize.py", coherent)]:
        (snapshot / name).write_bytes(p.read_bytes())
    actual_env = dict(PYTHONPATH=str(hook.parent), TIANSHU_TOPP_LAUNCH_PROBE="1",
                      TIANSHU_TOPP_LAUNCH_RECORD_DIR=str(root / "topp_launch_records"))
    original = subprocess.Popen

    def popen(command, *positional, **kwargs):
        if isinstance(command, list) and command[:2] == ["vllm", "serve"]:
            env = dict(kwargs["env"])
            env.update(actual_env)
            kwargs["env"] = env
        return original(command, *positional, **kwargs)

    spec = importlib.util.spec_from_file_location("_tianshu_topp_math_foundation", trial)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.Popen = popen
    sys.argv = [str(trial), "--output", str(root), "--candidate", "decode-coherent", "--service-root",
                "/workspace/benchmark-results/20261013_decode_coherent_service_tianshu",
                "--buckets", "1", "--gather", "1", "--bucket-trace", "2", "--max-batched-tokens", "2048"]
    try:
        code = module.main()
    finally:
        subprocess.Popen = original
    state = json.loads((root / "result.json").read_text())
    records = [json.loads(p.read_text()) for p in (root / "topp_launch_records").glob("route_*.json")]
    state.update(fresh_math_for_topp_launch=True, matched_topp_service_root=str(args.service_root),
                 actual_topp_environment=actual_env, topp_launch_records=records,
                 topp_hook=dict(path=str(hook), sha256=hashlib.sha256(hook.read_bytes()).hexdigest()))
    (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if code == 0:
        assert records and all(r["configuration"] == dict(num_warps=16, num_stages=1) for r in records)
        assert all(r["source_sha256"] == "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8" for r in records)
        assert state["installed_sources"] == service["installed_sources"]
        subprocess.run(["python3", str(base / "tianshu_decode_tiles_coherent_20261013/audit_math_eval.py"),
                        "--output", str(root)], check=True, capture_output=True)
        audit = json.loads((root / "accuracy_audit.json").read_text())
        audit.update(fresh_math_for_topp_launch=True, matched_topp_service_root=str(args.service_root),
                     actual_topp_environment=actual_env, topp_launch_records=records, topp_hook=state["topp_hook"])
        (root / "accuracy_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        subprocess.run(["python3", str(base / "archive_completed_pair.py"), "--single-probe", "--source", str(root),
                        "--completion-file", "accuracy_audit.json", "--label", "topp_launch_math_20261013"], check=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
