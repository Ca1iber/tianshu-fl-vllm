# SPDX-License-Identifier: Apache-2.0
"""Audit both-stage address widths on actual current .cv tensors and Graph events."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from address32 import render_source


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
    assert sha(base / "probe_address.py") == sha(snapshot / "probe_address.py") == data["probe_sha256"]
    for source in (base / "address32.py", base.parent / "tianshu_decode_cache_policy_20261014/experimental_cache.py",
                   base.parent / "tianshu_decode_low_warps_20261014/launch_router.py",
                   base.parent / "tianshu_native_prefill_20261012/zero_initialization.py"):
        assert source.read_bytes() == (snapshot / source.name).read_bytes()
    assert sha(snapshot / "triton_unified_attention.py") == "bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0"
    assert sha(root / "cache_policy_source/generated_cache_policy.py") == "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
    assert sha(root / "zero_initialization/generated_zero_initialization.py") == "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e"
    provenance = data["provenance"]
    assert not provenance["grid_changed"] and not provenance["installed_sources_modified"]
    assert provenance["producer_cache_modifier"] == ".cv"
    cv_source=root / "cache_policy_source/generated_cache_policy.py"
    narrow_source=root / "address32_source/cv_address32.py"
    assert narrow_source.read_bytes()==render_source(cv_source.read_bytes()).encode()
    generated_sha=sha(narrow_source)
    generation=provenance["address32_generation"]
    assert generation["generated_sha256"]==generated_sha
    assert generation["annotations_changed"]==28 and generation["casts_changed"]==6
    assert generation["floating_point_AST_unchanged"] and generation["pointer_types_unchanged"]
    candidate_module="_tianshu_cv_address32_"+generated_sha[:12]
    assert provenance["address_helper_sha256"] == sha(base / "address32.py")
    for name, digest in data["backend_source_sha256"].items(): assert sha(snapshot / name) == digest
    assert data["valid_address_modes"] == ["producer32", "both32"]
    expected = [(b, 4096) for b in (8, 16, 24, 32, 64)] + [(64, 5120), (64, 65)]
    expected += [(8, n) for n in (1, 15, 16, 17, 31, 32, 33, 63, 64, 65)]
    assert len(data["cases"]) == 17
    rows = []
    all_bitwise = True
    for case, (batch, length) in zip(data["cases"], expected):
        assert case["batch"] == batch and max(case["kv_lengths"]) == length
        assert case["logical_KV_identical"] and case["query_row_tiles"] == [16] * 3
        assert len(case["variants"]) == 2 and len(case["changed_graph_bitwise_equal"]) == 3
        all_bitwise &= all(case["changed_graph_bitwise_equal"])
        assert len(case["changed_graph_fp32_checks"]) == 3
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
        controls = [("control", case["baseline_launch_metadata"], case["baseline_raw_ms"], case["baseline_ms"])]
        for mode, variant in zip(("producer32", "both32"), case["variants"]):
            assert variant["address_mode"] == mode and variant["configuration"] == configuration
            assert variant["tails_preserved"]
            all_bitwise &= variant["candidate_bitwise_equal"]
            for key in ("candidate_vs_fp32", "candidate_vs_native"):
                assert variant[key]["passed"] and variant[key]["failed_elements"] == 0
            controls.append((mode, variant["launch_metadata"], variant["raw_ms"], variant["candidate_ms"]))
            change = 100 * (variant["candidate_ms"] / case["baseline_ms"] - 1)
            assert abs(change - variant["time_change_percent"]) < 1e-10
            rows.append(dict(batch=batch, kv_max=length, address_mode=mode,
                baseline_ms=case["baseline_ms"], candidate_ms=variant["candidate_ms"], time_change_percent=change,
                bitwise_equal=variant["candidate_bitwise_equal"]))
        for mode, metadata, timings, median in controls:
            assert metadata["num_warps"] == producer_warps and metadata["num_stages"] == 1
            assert metadata["actual_BLOCK_M"] == 16 and metadata["actual_BLOCK_Q"] == 2 and metadata["actual_BLOCK_SIZE"] == 16
            assert metadata["producer_and_reducer_tile"] == tile and metadata["requested_cache_modifier"] == ".cv"
            assert metadata["actual_grid"] == [queries // 2 + batch, 2, segments]
            reducer = metadata["actual_reducer"]
            assert reducer["actual_num_warps"] == 4
            assert metadata["address_mode"] == mode
            assert metadata["producer_source_module"] == ("tianshu_native_cache_policy" if mode=="control" else candidate_module)
            assert reducer["source_module"] == (candidate_module if mode=="both32" else "vllm.v1.attention.ops.triton_unified_attention")
            if mode=="control":assert metadata["address_bounds"]==[]
            else:assert metadata["address_bounds"] and all(0<=v<2**31 for v in metadata["address_bounds"])
            assert reducer["actual_num_stages"] == 3 and reducer["grid"] == [queries, 16]
            assert reducer["TILE_SIZE"] == tile and reducer["NUM_SEGMENTS_PER_SEQ"] == segments
            assert len(timings) == 9 and all(t > 0 for t in timings) and statistics.median(timings) == median
    artifacts=list((root/'kernel_artifacts').glob('*/manifest.json'))
    assert {p.parent.name for p in artifacts}=={'control','producer32','both32'}
    for path in artifacts:
        record=json.loads(path.read_text());assert record['files']
        assert record['requested_cache_modifier']=='.cv'
        for name,digest in record['files'].items():assert sha(path.parent/name)==digest
    output = dict(complete=True, official_benchmark=False, adopted=False, cases=17, candidate_cases=34,
                  all_original_and_changed_KV_bitwise_equal=all_bitwise,
                  recomputed_events=17 * 3 * 9, actual_producer_reducer_address_sources_verified=True, actual_tensor_offsets_bounded_below_signed32=True,
                  native_cv_and_narrow_source_bytes_verified=True, rows=rows)
    (root / "local_download_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}))


if __name__ == "__main__": main()
