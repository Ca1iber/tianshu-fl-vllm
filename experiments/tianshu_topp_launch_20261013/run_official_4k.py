# SPDX-License-Identifier: Apache-2.0
"""Original4K on source-matched TopP launch and verified coherent foundation."""
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
    parser.add_argument("--math-results", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    service = json.loads((args.service_root / "candidate/result.json").read_text())
    comparison = json.loads((args.service_root / "comparison.json").read_text())
    quality = json.loads((args.math_results / "accuracy_audit.json").read_text())
    assert comparison["complete"] and comparison["gain_percent"] >= 2
    assert min(comparison["candidate_rounds"][2:]) > max(comparison["control_rounds"][2:])
    assert quality["fresh_math_for_topp_launch"] and quality["official_requirement_passed"]
    assert quality["matched_topp_service_root"] == str(args.service_root)
    base = Path(__file__).resolve().parents[1]
    foundation = base / "tianshu_decode_tiles_coherent_20261013"
    hook = Path(__file__).with_name("hooks") / "sitecustomize.py"
    sha = hashlib.sha256(hook.read_bytes()).hexdigest()
    assert sha == service["topp_hook_sha256"] == quality["topp_hook"]["sha256"]
    assert hook.read_bytes() == (args.math_results / "topp_source_snapshot/topp_sitecustomize.py").read_bytes()
    root.mkdir(parents=True)
    snapshot = root / "topp_source_snapshot"
    snapshot.mkdir()
    (snapshot / "topp_sitecustomize.py").write_bytes(hook.read_bytes())
    (snapshot / "run_official_4k.py").write_bytes(Path(__file__).read_bytes())
    actual_env = dict(PYTHONPATH=str(hook.parent), TIANSHU_TOPP_LAUNCH_PROBE="1",
                      TIANSHU_TOPP_LAUNCH_RECORD_DIR=str(root / "topp_launch_records"))
    original = subprocess.Popen

    def popen(command, *positional, **kwargs):
        if isinstance(command, list) and command[:2] == ["vllm", "serve"]:
            env = dict(kwargs["env"])
            env.update(actual_env)
            kwargs["env"] = env
        return original(command, *positional, **kwargs)

    trial = foundation / "run_official_4k.py"
    spec = importlib.util.spec_from_file_location("_topp_official_foundation", trial)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.Popen = popen
    # Foundation validators verify the unchanged Decode/Prefill computation;
    # this wrapper independently gates and audits the new TopP source/routes.
    foundation_service = "/workspace/benchmark-results/20261013_decode_coherent_service_tianshu"
    sys.argv = [str(trial), "--output", str(root), "--control-trial", foundation_service + "/control",
                "--candidate-trial", foundation_service + "/candidate", "--math-results", str(args.math_results),
                "--max-batched-tokens", "2048", "--decode-launch", "1", "--buckets", "1", "--gather", "1",
                "--bucket-trace", "2"]
    try:
        code = module.main()
    finally:
        subprocess.Popen = original
    state = json.loads((root / "result.json").read_text())
    records = [json.loads(p.read_text()) for p in (root / "topp_launch_records").glob("route_*.json")]
    state.update(fresh_math_for_topp_launch=True, matched_topp_service_root=str(args.service_root),
                 actual_topp_environment=actual_env, topp_launch_records=records, topp_hook=quality["topp_hook"],
                 short_gain_percent=comparison["gain_percent"])
    (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if code == 0:
        assert records and all(r["configuration"] == dict(num_warps=16, num_stages=1) for r in records)
        assert {r["source_sha256"] for r in records} == {r["source_sha256"] for r in quality["topp_launch_records"]}
        subprocess.run(["python3", str(foundation / "audit_official.py"), "--output", str(root),
                        "--math-results", str(args.math_results), "--previous-best-tok-s", "3426.62",
                        "--previous-best-ttft-ms", "4514.45"], check=True, capture_output=True)
        audit = json.loads((root / "comparison.json").read_text())
        audit.update(fresh_math_for_topp_launch=True, matched_topp_service_root=str(args.service_root),
                     topp_launch_records=records, topp_hook=quality["topp_hook"])
        audit["profile"] += " +nativeTopP w16/stage1, body/params/RNG unchanged"
        (root / "comparison.json").write_text(json.dumps(audit, indent=2) + "\n")
        subprocess.run(["python3", str(base / "archive_completed_pair.py"), "--single-probe", "--source", str(root),
                        "--label", "topp_launch_official4096_20261013"], check=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
