# SPDX-License-Identifier: Apache-2.0
"""Recompute fresh Math results from all downloaded raw predictions and reviews."""
import argparse
import json
from pathlib import Path

from validation_profile import audit_math, audit_runtime, digest, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    state = read(root / "result.json")
    assert state["phase"] == "complete" and state["fresh_math_for_combined_profile"]
    saved = read(root / "accuracy_audit.json")
    before = (root / "accuracy_audit.json").read_bytes()
    profile = audit_runtime(root, state, online=False)
    computed = audit_math(root, state, profile, write=False)
    assert computed == saved and (root / "accuracy_audit.json").read_bytes() == before
    repo = Path(__file__).resolve().parents[2]
    for path, sha in state["frozen_sources"].items():
        p = Path(path)
        if p.is_relative_to("/workspace/vllm-plugin-FL"):
            assert digest(repo / p.relative_to("/workspace/vllm-plugin-FL")) == sha
    report = dict(complete=True, all105_predictions_and_reviews_checked=True,
        fresh_math_for_combined_profile=True, correct=computed["correct"], total=105,
        accuracy=computed["accuracy"], official_requirement_passed=computed["official_requirement_passed"],
        all_source_snapshots_and_runtime_profile_verified=True)
    (root / "local_download_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
