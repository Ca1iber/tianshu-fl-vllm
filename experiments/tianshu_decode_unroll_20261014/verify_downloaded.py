# SPDX-License-Identifier: Apache-2.0
"""Audit loop-unroll hints with raw events and actual generated .cv source bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from unroll import render_source


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
    assert sha(base / "probe_unroll.py") == sha(snapshot / "probe_unroll.py") == data["probe_sha256"]
    for source in (base / "unroll.py", base.parent / "tianshu_decode_cache_policy_20261014/experimental_cache.py",
                   base.parent / "tianshu_decode_low_warps_20261014/launch_router.py",
                   base.parent / "tianshu_native_prefill_20261012/zero_initialization.py"):
        assert source.read_bytes() == (snapshot / source.name).read_bytes()
    assert sha(snapshot / "triton_unified_attention.py") == "bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0"
    assert sha(root / "cache_policy_source/generated_cache_policy.py") == "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
    assert sha(root / "zero_initialization/generated_zero_initialization.py") == "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e"
    provenance = data["provenance"]
    assert not provenance["grid_changed"] and not provenance["installed_sources_modified"]
    generation=provenance["loop_unroll_generation"]
    assert generation["arithmetic_body_and_reducer_source_unchanged"]
    assert generation["only_loop_header_and_constexpr_added"]
    cv=root/"cache_policy_source/generated_cache_policy.py"
    unrolled=root/"unroll_source/generated_loop_unroll.py"
    assert unrolled.read_bytes()==render_source(cv.read_bytes()).encode()
    assert sha(unrolled)==generation["generated_sha256"]
    assert sha(snapshot/"language_core.py")==provenance["language_core_sha256"]
    assert "loop_unroll_factor" in (snapshot/"language_core.py").read_text()
    assert provenance["unroll_helper_sha256"] == sha(base / "unroll.py")
    for name, digest in data["backend_source_sha256"].items(): assert sha(snapshot / name) == digest
    assert data["valid_loop_unroll_factors"] == [1, 2, 4, 8]
    expected = [(b, 4096) for b in (8, 16, 24, 32, 64)] + [(64, 5120), (64, 65)]
    expected += [(8, n) for n in (1, 15, 16, 17, 31, 32, 33, 63, 64, 65)]
    assert len(data["cases"]) == 17
    rows = []
    all_bitwise = True
    for case, (batch, length) in zip(data["cases"], expected):
        assert case["batch"] == batch and max(case["kv_lengths"]) == length
        assert case["logical_KV_identical"] and case["query_row_tiles"] == [16] * 5
        assert len(case["variants"]) == 4 and len(case["changed_graph_bitwise_equal"]) == 5
        all_bitwise &= all(case["changed_graph_bitwise_equal"])
        assert len(case["changed_graph_fp32_checks"]) == 5
        for group in ("cpu_reference_checks", "changed_graph_cpu_checks", "changed_graph_fp32_checks"):
            assert case[group] and all(x["passed"] and x["failed_elements"] == 0 and x["atol"] == .02 and x["rtol"] == .01 for x in case[group])
        assert case["original_vs_fp32"]["passed"] and case["baseline_tail_preserved"]
        queries = sum(case["query_lengths"])
        segments = 8 if batch == 64 else 4 if batch >= 16 else 16
        tile = 16 if batch in (8, 16, 32, 40, 48, 56, 64) else 32
        producer_warps = 1 if tile == 16 else 2
        configuration = dict(TILE_SIZE=tile, num_warps=producer_warps, num_stages=1, KV_CACHE_MODIFIER=".cv")
        assert case["baseline_configuration"] == configuration
        assert case["segments"] == segments
        controls = [(None, case["baseline_launch_metadata"], case["baseline_raw_ms"], case["baseline_ms"])]
        for factor, variant in zip((1, 2, 4, 8), case["variants"]):
            assert variant["configuration"] == dict(configuration,KV_LOOP_UNROLL=factor)
            assert variant["tails_preserved"]
            all_bitwise &= variant["candidate_bitwise_equal"]
            for key in ("candidate_vs_fp32", "candidate_vs_native"):
                assert variant[key]["passed"] and variant[key]["failed_elements"] == 0
            controls.append((factor, variant["launch_metadata"], variant["raw_ms"], variant["candidate_ms"]))
            change = 100 * (variant["candidate_ms"] / case["baseline_ms"] - 1)
            assert abs(change - variant["time_change_percent"]) < 1e-10
            rows.append(dict(batch=batch, kv_max=length, loop_unroll_factor=factor,
                baseline_ms=case["baseline_ms"], candidate_ms=variant["candidate_ms"], time_change_percent=change,
                bitwise_equal=variant["candidate_bitwise_equal"]))
        for requested, metadata, timings, median in controls:
            assert metadata["num_warps"] == producer_warps and metadata["num_stages"] == 1
            assert metadata["actual_BLOCK_M"] == 16 and metadata["actual_BLOCK_Q"] == 2 and metadata["actual_BLOCK_SIZE"] == 16
            assert metadata["producer_and_reducer_tile"] == tile and metadata["requested_cache_modifier"] == ".cv"
            assert metadata["actual_grid"] == [queries // 2 + batch, 2, segments]
            assert metadata['requested_unroll_factor']==requested
            assert len(timings) == 9 and all(t > 0 for t in timings) and statistics.median(timings) == median
    artifacts=list((root/'kernel_artifacts').glob('*/manifest.json'))
    assert {p.parent.name for p in artifacts}=={'baseline_cv','1','2','4','8'}
    for path in artifacts:
        record=json.loads(path.read_text());assert record['files']
        assert record['requested_cache_modifier']=='.cv'
        expected=None if path.parent.name=='baseline_cv' else int(path.parent.name)
        assert record['register_info']['requested_unroll_factor']==expected
        for name,digest in record['files'].items():assert sha(path.parent/name)==digest
    output = dict(complete=True, official_benchmark=False, adopted=False, cases=17, candidate_cases=68,
                  all_original_and_changed_KV_bitwise_equal=all_bitwise,
                  recomputed_events=17 * 5 * 9, actual_producer_unroll_parameters_and_unchanged_configuration_verified=True, actual_lowered_unrolling_not_claimed=True,
                  native_cv_and_unroll_source_bytes_verified=True, rows=rows)
    (root / "local_download_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}))


if __name__ == "__main__": main()
