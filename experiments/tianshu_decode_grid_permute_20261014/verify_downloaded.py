# SPDX-License-Identifier: Apache-2.0
"""Audit complete CTA-permutation raw events, source bytes and changed-KV checks."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics

from grid_permute import CV_SHA, permute_grid, render_source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = json.loads((root / "result.json").read_text())
    assert data["complete"] and data["numerical_checks_passed"]
    assert not data["official_benchmark"] and not data["adoption_allowed"]
    assert not data.get("compile_errors") and not data.get("changed_input_failures")
    base = Path(__file__).resolve().parent
    snapshot = root / "source_snapshot"
    assert sha(base / "probe_grid.py") == sha(snapshot / "probe_grid.py") == data["probe_sha256"]
    for local in (base / "grid_permute.py", base.parent / "tianshu_decode_cache_policy_20261014/experimental_cache.py",
                  base.parent / "tianshu_decode_low_warps_20261014/launch_router.py",
                  base.parent / "tianshu_native_prefill_20261012/zero_initialization.py"):
        assert local.read_bytes() == (snapshot / local.name).read_bytes()
    assert sha(snapshot / "triton_unified_attention.py") == "bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0"
    cv = root / "cache_policy_source/generated_cache_policy.py"
    assert sha(cv) == CV_SHA
    permutations = list(itertools.permutations(range(3)))
    assert data["valid_axis_permutations"] == [list(axes) for axes in permutations]
    provenance = data["provenance"]
    assert provenance["grid_changed"] and not provenance["installed_sources_modified"]
    assert not provenance["tile_reduction_order_may_change"]
    assert provenance["grid_helper_sha256"] == sha(base / "grid_permute.py")
    assert len(provenance["grid_generations"]) == 6
    for axes, record in zip(permutations, provenance["grid_generations"]):
        assert record["logical_to_physical_axes"] == list(axes) and record["base_cv_sha256"] == CV_SHA
        assert not record["attention_arithmetic_changed"] and not record["reducer_changed"]
        source = root / "grid_sources" / "".join(map(str, axes)) / "generated_grid_permute.py"
        assert source.read_bytes() == render_source(cv.read_bytes(), axes).encode()
        assert sha(source) == record["generated_sha256"]
    expected = [(b, 4096) for b in (8, 16, 24, 32, 64)] + [(64, 5120), (64, 65)]
    expected += [(8, n) for n in (1, 15, 16, 17, 31, 32, 33, 63, 64, 65)]
    assert len(data["cases"]) == len(expected)
    rows = []
    for case, (batch, length) in zip(data["cases"], expected):
        assert case["batch"] == batch and max(case["kv_lengths"]) == length
        assert case["logical_KV_identical"] and case["query_row_tiles"] == [16] * 7
        assert len(case["variants"]) == 6
        assert len(case["changed_graph_bitwise_equal"]) == 7 and all(case["changed_graph_bitwise_equal"])
        assert len(case["changed_graph_fp32_checks"]) == 7
        for group in ("cpu_reference_checks", "changed_graph_cpu_checks", "changed_graph_fp32_checks"):
            assert case[group] and all(x["passed"] and x["failed_elements"] == 0 and x["atol"] == .02 and x["rtol"] == .01 for x in case[group])
        assert case["original_vs_fp32"]["passed"] and case["baseline_tail_preserved"]
        queries = sum(case["query_lengths"])
        segments = 8 if batch == 64 else 4 if batch >= 16 else 16
        assert case["segments"] == segments
        logical = (queries // 2 + batch, 2, segments)
        tile = 16 if batch in (8, 16, 32, 40, 48, 56, 64) else 32
        warps = 1 if tile == 16 else 2
        assert case["baseline_configuration"] == dict(TILE_SIZE=tile, num_warps=warps, num_stages=1, KV_CACHE_MODIFIER=".cv")
        controls = [(None, case["baseline_launch_metadata"], case["baseline_raw_ms"], case["baseline_ms"])]
        for axes, variant in zip(permutations, case["variants"]):
            assert variant["logical_to_physical_axes"] == list(axes)
            assert variant["configuration"] == case["baseline_configuration"]
            assert variant["candidate_bitwise_equal"] and variant["tails_preserved"]
            for key in ("candidate_vs_fp32", "candidate_vs_native"):
                assert variant[key]["passed"] and variant[key]["failed_elements"] == 0
            controls.append((axes, variant["launch_metadata"], variant["raw_ms"], variant["candidate_ms"]))
            change = 100 * (variant["candidate_ms"] / case["baseline_ms"] - 1)
            assert abs(change - variant["time_change_percent"]) < 1e-10
            rows.append(dict(batch=batch, kv_max=length, axes=axes,
                             baseline_ms=case["baseline_ms"], candidate_ms=variant["candidate_ms"], time_change_percent=change))
        for axes, metadata, timings, median in controls:
            assert metadata["num_warps"] == warps and metadata["num_stages"] == 1
            assert metadata["actual_BLOCK_M"] == 16 and metadata["actual_BLOCK_Q"] == 2 and metadata["actual_BLOCK_SIZE"] == 16
            assert metadata["producer_and_reducer_tile"] == tile and metadata["requested_cache_modifier"] == ".cv"
            assert metadata["logical_grid"] == list(logical)
            assert metadata["logical_to_physical_axes"] == (None if axes is None else list(axes))
            assert metadata["actual_grid"] == list(logical if axes is None else permute_grid(logical, axes))
            assert len(timings) == 9 and all(t > 0 for t in timings) and statistics.median(timings) == median
    artifacts = list((root / "kernel_artifacts").glob("*/manifest.json"))
    assert len(artifacts) == 7
    for path in artifacts:
        artifact = json.loads(path.read_text())
        assert artifact["requested_cache_modifier"] == ".cv" and artifact["files"]
        for name, digest in artifact["files"].items():
            assert sha(path.parent / name) == digest
    output = dict(complete=True, official_benchmark=False, adopted=False, cases=17, candidate_cases=102,
                  original_and_changed_KV_bitwise_checks_passed=True, recomputed_events=17 * 7 * 9,
                  generated_source_bytes_reproduced=True, rows=rows)
    (root / "local_download_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}))


if __name__ == "__main__":
    main()
