# SPDX-License-Identifier: Apache-2.0
"""Audit asymmetric K/V load hints, source bytes, raw events, and changed-input correctness."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from cache_policy import render_source, POLICIES, configuration


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


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
    assert sha(base / "probe_cache.py") == sha(snapshot / "probe_cache.py") == data["probe_sha256"]
    for source in (base / "cache_policy.py", base.parent / "tianshu_decode_cache_policy_20261014/experimental_cache.py",
                   base.parent / "tianshu_decode_low_warps_20261014/launch_router.py",
                   base.parent / "tianshu_native_prefill_20261012/zero_initialization.py"):
        assert source.read_bytes() == (snapshot / source.name).read_bytes()
    assert sha(snapshot / "triton_unified_attention.py") == "bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0"
    assert sha(root / "cache_policy_source/generated_cache_policy.py") == "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
    assert sha(root / "zero_initialization/generated_zero_initialization.py") == "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e"
    provenance = data["provenance"]
    assert not provenance["grid_changed"] and not provenance["installed_sources_modified"]
    generation = provenance["asymmetric_generation"]
    assert generation["attention_arithmetic_unchanged"]
    assert generation["exactly_reversible_to_current_CV_source"]
    assert generation["only_3D_key_value_load_hints_changed"]
    cv = root / "cache_policy_source/generated_cache_policy.py"
    asymmetric = root / "asymmetric_source/generated_asymmetric_cache.py"
    assert asymmetric.read_bytes() == render_source(cv.read_bytes()).encode()
    assert sha(asymmetric) == generation["generated_sha256"]
    assert provenance["asymmetric_helper_sha256"] == sha(base / "cache_policy.py")
    for name, digest in data["backend_source_sha256"].items():
        assert sha(snapshot / name) == digest
    assert data["valid_strict_bitwise_policies"] == [list(p) for p in POLICIES]
    state = json.loads((root.parent / "state.json").read_text())
    assert state["phase"] == "complete" and state["gpu_idle_after"]
    assert len(state["frozen_sources"]) == 6
    for source, digest in state["frozen_sources"].items():
        relative = Path(source).relative_to("/workspace/vllm-plugin-FL/experiments")
        assert sha(base.parent / relative) == sha(root.parent / "source_snapshot" / relative) == digest
    expected = [(b, 4096) for b in (8, 16, 24, 32, 64)] + [(64, 5120), (64, 65)]
    expected += [(8, n) for n in (1, 15, 16, 17, 31, 32, 33, 63, 64, 65)]
    assert len(data["cases"]) == 17
    rows = []
    all_bitwise = True
    for case, (batch, length) in zip(data["cases"], expected):
        assert case["batch"] == batch and max(case["kv_lengths"]) == length
        assert case["logical_KV_identical"] and case["query_row_tiles"] == [16] * 10
        assert len(case["variants"]) == 9 and len(case["changed_graph_bitwise_equal"]) == 10
        all_bitwise &= all(case["changed_graph_bitwise_equal"])
        assert len(case["changed_graph_fp32_checks"]) == 10
        for group in ("cpu_reference_checks", "changed_graph_cpu_checks", "changed_graph_fp32_checks"):
            assert case[group] and all(x["passed"] and x["failed_elements"] == 0 and x["atol"] == .02 and x["rtol"] == .01 for x in case[group])
        assert case["original_vs_fp32"]["passed"] and case["baseline_tail_preserved"]
        queries = sum(case["query_lengths"])
        segments = 8 if batch == 64 else 4 if batch >= 16 else 16
        tile = 16 if batch in (8, 16, 32, 40, 48, 56, 64) else 32
        producer_warps = 1 if tile == 16 else 2
        common = dict(TILE_SIZE=tile, num_warps=producer_warps, num_stages=1, KV_CACHE_MODIFIER=".cv")
        assert case["baseline_configuration"] == common
        assert case["segments"] == segments
        controls = [(None, case["baseline_launch_metadata"], case["baseline_raw_ms"], case["baseline_ms"])]
        for policy, variant in zip(POLICIES, case["variants"]):
            assert variant["configuration"] == dict(common, **configuration(policy))
            assert variant["tails_preserved"]
            all_bitwise &= variant["candidate_bitwise_equal"]
            for key in ("candidate_vs_fp32", "candidate_vs_native"):
                assert variant[key]["passed"] and variant[key]["failed_elements"] == 0
            controls.append((configuration(policy), variant["launch_metadata"], variant["raw_ms"], variant["candidate_ms"]))
            change = 100 * (variant["candidate_ms"] / case["baseline_ms"] - 1)
            assert abs(change - variant["time_change_percent"]) < 1e-10
            rows.append(dict(batch=batch, kv_max=length, policy=list(policy),
                baseline_ms=case["baseline_ms"], candidate_ms=variant["candidate_ms"], time_change_percent=change,
                bitwise_equal=variant["candidate_bitwise_equal"]))
        for requested, metadata, timings, median in controls:
            assert metadata["num_warps"] == producer_warps and metadata["num_stages"] == 1
            assert metadata["actual_BLOCK_M"] == 16 and metadata["actual_BLOCK_Q"] == 2 and metadata["actual_BLOCK_SIZE"] == 16
            assert metadata["producer_and_reducer_tile"] == tile and metadata["requested_cache_modifier"] == ".cv"
            assert metadata["actual_grid"] == [queries // 2 + batch, 2, segments]
            assert metadata['requested_asymmetric_policy'] == (requested if requested is not None else {k: None for k in configuration(POLICIES[0])})
            assert len(timings) == 9 and all(t > 0 for t in timings) and statistics.median(timings) == median
    artifacts = list((root / "kernel_artifacts").glob("*/manifest.json"))
    tags = {"baseline_cv": {k: None for k in configuration(POLICIES[0])}}
    for policy in POLICIES:
        tag = "_".join(x or "default" for x in policy).replace(".", "")
        tags[tag] = configuration(policy)
    assert {p.parent.name for p in artifacts} == set(tags)
    for path in artifacts:
        record = json.loads(path.read_text())
        assert record["files"]
        assert record["requested_cache_modifier"] == ".cv"
        assert record["register_info"]["requested_asymmetric_policy"] == tags[path.parent.name]
        for name, digest in record["files"].items():
            assert sha(path.parent / name) == digest
    assert all_bitwise
    output = dict(complete=True, official_benchmark=False, adopted=False, cases=17, candidate_cases=153,
                  all_original_and_changed_KV_bitwise_equal=all_bitwise,
                  recomputed_events=17 * 10 * 9,
                  actual_producer_parameters_and_unchanged_configuration_verified=True,
                  native_cv_and_asymmetric_source_bytes_verified=True, frozen_sources_verified=6, rows=rows)
    (root / "local_download_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}))


if __name__ == "__main__": main()
