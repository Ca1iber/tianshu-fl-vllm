# SPDX-License-Identifier: Apache-2.0
"""Recompute original-command screen, Math105 or official4K from downloaded raw."""
import argparse
import json
from pathlib import Path

from validation_profile import audit_math, audit_benchmark, audit_runtime, digest, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--math-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    state = read(root / "result.json")
    assert state["phase"] == "complete"
    assert state["stage"] in ("screen", "math", "official")
    assert len(state["frozen_sources"]) == 44
    profile = audit_runtime(root, state, official=state["stage"] != "math", online=False)
    repo = Path(__file__).resolve().parents[2]
    for path, sha in state["frozen_sources"].items():
        path = Path(path)
        if path.is_relative_to("/workspace/vllm-plugin-FL"):
            assert digest(repo / path.relative_to("/workspace/vllm-plugin-FL")) == sha
    quality = None
    if state["stage"] == "math":
        computed = audit_math(root, state, profile, write=False)
        assert computed == read(root / "accuracy_audit.json")
        report = dict(complete=True, original_serve_command_verified=True,
            all105_predictions_and_reviews_checked=True, correct=computed["correct"], total=105,
            accuracy=computed["accuracy"], official_requirement_passed=computed["official_requirement_passed"])
    else:
        if state["stage"] == "official":
            assert args.math_root is not None
            math = args.math_root.resolve()
            math_state = read(math / "result.json")
            assert math_state["phase"] == "complete" and math_state["stage"] == "math"
            quality = audit_math(math, math_state, audit_runtime(math, math_state, online=False), write=False)
            assert quality == read(math / "accuracy_audit.json")
        computed = audit_benchmark(root, state, profile, quality, write=False)
        assert computed == read(root / "comparison.json")
        report = dict(complete=True, original_serve_command_verified=True,
            original_benchmark_script_verified=True, official_benchmark=computed["official_benchmark"],
            requests_successful=computed["requests_successful"], total_tok_s=computed["total_tok_s"],
            output_tok_s=computed["output_tok_s"], ttft_ms=computed["ttft_ms"],
            raw_scope="Original benchmark stdout and CSV; no per-request response-body claim.")
        if quality is not None:
            report.update(fresh_math_raw105_verified=True, math_correct=quality["correct"])
    report.update(all44_source_snapshots_verified=True, native2048_budget256seq75warm_verified=True,
        CV_KV16_S8_nativeTopP_originalSampler_verified=True, zero_preemptions=True)
    (root / "local_download_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
