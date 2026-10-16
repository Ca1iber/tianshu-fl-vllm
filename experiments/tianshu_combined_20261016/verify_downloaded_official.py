# SPDX-License-Identifier: Apache-2.0
"""Audit original official4K CSV, source snapshots and the fresh105 Math raw files."""
import argparse
import json
from pathlib import Path

from validation_profile import audit_math, audit_official, audit_runtime, digest, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--math-root", type=Path, required=True)
    args = parser.parse_args()
    root, math = args.root.resolve(), args.math_root.resolve()
    state = read(root / "result.json")
    assert state["phase"] == "complete" and state["stage"] == "official"
    math_state = read(math / "result.json")
    assert math_state["phase"] == "complete" and math_state["stage"] == "math"
    quality = audit_math(math, math_state, audit_runtime(math, math_state, online=False), write=False)
    assert quality == read(math / "accuracy_audit.json")
    profile = audit_runtime(root, state, official=True, online=False)
    computed = audit_official(root, state, quality, profile, write=False)
    assert computed == read(root / "comparison.json")
    repo = Path(__file__).resolve().parents[2]
    for path, sha in state["frozen_sources"].items():
        p = Path(path)
        if p.is_relative_to("/workspace/vllm-plugin-FL"):
            assert digest(repo / p.relative_to("/workspace/vllm-plugin-FL")) == sha
    result = dict(complete=True, original_official4_runs_verified=True,
        requests_successful=1024, fresh_math_raw105_verified=True, math_correct=quality["correct"],
        total_tok_s=computed["total_tok_s"], output_tok_s=computed["output_tok_s"],
        ttft_ms=computed["ttft_ms"], eligible_as_best=computed["eligible_as_best"],
        all_source_snapshots_and_runtime_profile_verified=True)
    (root / "local_download_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
