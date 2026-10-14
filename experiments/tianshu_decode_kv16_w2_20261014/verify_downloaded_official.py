# SPDX-License-Identifier: Apache-2.0
"""Verify original official4K CSV aggregates and matching independent Math."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--math-root", type=Path, required=True)
    args = parser.parse_args()
    root, math = args.root.resolve(), args.math_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    state = json.loads((root / "result.json").read_text())
    result = json.loads((root / "comparison.json").read_text())
    quality = json.loads((math / "accuracy_audit.json").read_text())
    assert state["phase"] == "complete" and state["gpu_idle_after"]
    assert state["benchmark_exit_code"] == 0
    assert state["completed_runs"] == state["runs"] == 4 and state["skip_first"] == 1
    assert result["complete"] and result["official_benchmark"]
    assert quality["complete"] and quality["total"] == 105 and quality["accuracy"] > .95
    assert quality["fresh_math_for_decode_kv16_w1"] and not quality["reused_original_accuracy"]
    assert state["candidate_hook"] == quality["candidate_hook"]
    assert state["experimental_kernel"] == quality["experimental_kernel"]
    assert state["server_command"] == quality["server_command"]
    assert state["kv16_sources"] == quality["kv16_sources"]
    raw_paths = list((root / "benchmark_results").glob("raw_runs_*.csv"))
    summary_paths = list((root / "benchmark_results").glob("summary_*.csv"))
    assert len(raw_paths) == len(summary_paths) == 1
    with raw_paths[0].open(newline="") as f:
        rows = list(csv.DictReader(f))
    with summary_paths[0].open(newline="") as f:
        summary = list(csv.DictReader(f))
    assert len(rows) == 4 and len(summary) == 1
    assert rows == state["raw_runs"] and summary[0] == state["official_summary"]
    for row in rows:
        assert row["Run Status"] == "SUCCESS" and float(row["Successful Requests"]) == 256
        assert [int(row[k]) for k in ("Prefill", "Decode", "Conc", "Num Prompts")] == [4096, 1024, 64, 256]
        assert float(row["Total Input Tokens"]) == 1048576 and float(row["Total Output Tokens"]) == 262144
    for column in ("Total tok/s", "Output tok/s", "Mean TTFT (ms)"):
        assert abs(float(summary[0][column]) - statistics.mean(float(r[column]) for r in rows[1:])) < .011
    assert float(summary[0]["Total tok/s"]) == result["total_tok_s"]
    assert float(summary[0]["Mean TTFT (ms)"]) == result["ttft_ms"]
    assert result["math_correct"] == quality["correct"]
    assert result["previous_best_total_tok_s"] == 3640.16
    assert result["adopt_as_best"] == (result["total_tok_s"] > 3640.16)
    assert result["excluded_runs"] == [1]
    benchmark = root / "source_snapshot/benchmark_throughput_serve.py"
    assert digest(benchmark) == state["official_script_sha256"] == "0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565"
    for name in ("run_official_4k.py", "audit_official.py", "sitecustomize.py", "decode_sitecustomize.py"):
        relative = "hooks/" + name if "sitecustomize" in name else name
        assert (root / "official_source_snapshot" / name).read_bytes() == (Path(__file__).resolve().parent / relative).read_bytes()
    for path, sha in quality["kv16_sources"].items():
        relative = Path(path).relative_to("/workspace/vllm-plugin-FL")
        assert digest(repo / relative) == sha
        assert digest(root / "source_snapshot" / ("kv16_" + Path(path).name)) == sha
    for name, meta in quality["installed_sources"].items():
        assert digest(root / "source_snapshot" / name) == meta["sha256"]
    routes = state["candidate_routing_records"]
    assert routes and state["fresh_math_for_decode_kv16_w1"]
    for x in routes:
        tile = 16 if x["batch"] in (8,16,32,40,48,56,64) else 32
        warps = 1 if tile == 16 else 2 if 8 <= x["batch"] <= 64 else 4
        assert x["configuration"] == dict(TILE_SIZE=tile, num_warps=warps, num_stages=1)
        assert x["compiled_metadata"]["num_warps"] == warps
        assert x["generated_source_sha256"] == "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e"
    assert any(x["batch"] == 64 and x["configuration"] == dict(TILE_SIZE=16,num_warps=1,num_stages=1) for x in routes)
    reducers = state["decode_coherent_reducer_records"]
    assert {(x["batch"], x["segments"]) for x in routes} == {(x["batch"], x["segments"]) for x in reducers}
    assert reducers and all(x["actual_tile"] == (16 if x["batch"] in (8,16,32,40,48,56,64) else 32) for x in reducers)
    topp = state["topp_launch_records"]
    assert all(x["configuration"] == dict(num_warps=16, num_stages=1) and x["compiled_metadata"]["num_warps"] == 16 for x in topp)
    assert any(x["complete"] and len(x["calls"]) == 75 for x in state["bucket_warmup_records"])
    assert any(x["index_dtype"] == "torch.int32" for x in state["actual_gather_records"])
    assert "No running processes found" in (root / "hardware_after.txt").read_text()
    verified = dict(complete=True, original_official_csv_runs_verified=4,
                    successful_requests_from_original_aggregates=1024, request_validation_scope="Original official CSV aggregate counts and token totals",
                    excluded_runs=[1], summary_means_verified=True, math_correct=quality["correct"], math_total=105,
                    actual_sources_and_cli_and_routes_verified=True, total_tok_s=result["total_tok_s"],
                    output_tok_s=result["output_tok_s"], ttft_ms=result["ttft_ms"], adopt_as_best=result["adopt_as_best"])
    (root / "local_download_audit.json").write_text(json.dumps(verified, indent=2) + "\n")
    print(json.dumps(verified))


if __name__ == "__main__":
    main()
