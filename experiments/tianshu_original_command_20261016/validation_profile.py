# SPDX-License-Identifier: Apache-2.0
"""Source and runtime evidence for fresh validation of the original-command profile."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics


CV_SHA = "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
BENCH_SHA = "0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565"


def official_command():
    return ["vllm", "serve", "/workspace/MiniCPM5-2B", "--port", "9031",
        "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        "--served-model-name", "minicpm", "--gpu-memory-utilization", "0.85",
        "--max-model-len", "131072"]


def prepare(root, reference_root):
    """Reuse only measured operator switches; leave engine parameters at defaults."""
    base = Path(__file__).resolve().parent
    template = read(base / "profile_environment_template.json")
    old_root = template["old_output_root"]
    selected = template["environment"]
    assert len(selected) == 35 and selected["TIANSHU_CACHE_SERVICE_MODE"] == "candidate"
    assert selected["TIANSHU_MAX_SEQS_MODE"] == "control"
    mapped = {key: value.replace(old_root + "/", str(root) + "/") for key, value in selected.items()}
    assert all(old_root not in value for value in mapped.values())
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith(("TIANSHU_", "VLLM_FL_", "FLAGGEMS_ILUVATAR_")) or key in {
            "VLLM_PLUGINS", "PYTHONPATH", "PYTHONUNBUFFERED", "TRITON_PRINT_AUTOTUNING", "CUDA_VISIBLE_DEVICES"
        }:
            del environment[key]
    environment.update(mapped)
    reference = read(reference_root / "control/result.json")
    assert reference["actual_cache_mode_verified"] and reference["cache_modifier"] == ".cv"
    sources = dict(read(reference_root / "comparison.json")["frozen_sources"])
    repo = base.parents[1]
    for relative in [
        "tianshu_buckets75_recheck_20261012/joint_hooks/sitecustomize.py",
        "tianshu_buckets75_recheck_20261012/hooks/sitecustomize.py",
        "tianshu_decode_launch_20261012/hooks/sitecustomize.py",
        "tianshu_native_prefill_20261012/hooks/sitecustomize.py",
        "tianshu_native_prefill_20261012/zero_initialization.py",
        "tianshu_fa2_tiles_20261012/hooks/sitecustomize.py",
    ]:
        path = repo / "experiments" / relative
        sources[str(path)] = digest(path)
    for metadata in list(reference["installed_sources"].values()) + list(reference["installed_sampling_sources"].values()):
        sources[metadata["path"]] = metadata["sha256"]
    attention = "/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py"
    sources[attention] = "bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0"
    sources["/usr/local/lib/python3.12/site-packages/vllm/engine/arg_utils.py"] = "f1a32e2207e9ed2f0d954e3e6457effca17ca907cb0cf17bba2ca8bde855a719"
    sources["/usr/local/lib/python3.12/site-packages/vllm/config/scheduler.py"] = "4102ccd3d73188200b25f47a0cba837853662647850f283012d8cc1f9e8c47c2"
    for path in (Path(__file__), base / "run_validation.py", base / "profile_environment_template.json"):
        sources[str(path)] = digest(path)
    benchmark = repo / "benchmarks/benchmark_throughput_serve.py"
    assert digest(benchmark) == BENCH_SHA
    sources[str(benchmark)] = BENCH_SHA
    frozen_check(sources)
    for path, sha in sources.items():
        destination = root / "source_snapshot" / Path(path).relative_to("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(path).read_bytes())
        assert digest(destination) == sha
    return official_command(), environment, mapped, sources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_check(sources):
    assert sources and all(digest(path) == sha for path, sha in sources.items())






def records(root, folder, pattern):
    return [read(p) for p in sorted((root / folder).glob(pattern))]


def audit_runtime(root, state, official=False, online=True):
    if online:
        frozen_check(state["frozen_sources"])
    original_root = Path(state["output_root"])

    def artifact(path):
        return root / Path(path).relative_to(original_root)

    for path, sha in state["frozen_sources"].items():
        assert digest(root / "source_snapshot" / Path(path).relative_to("/")) == sha
    workers = records(root, "max_seqs_records", "worker_*.json")
    assert workers and all(r["mode"] == "control" and r["max_num_seqs"] == 256 and
        r["async_scheduling"] is True and r["max_num_batched_tokens"] == 2048 and
        r["model_max_len"] == 131072 for r in workers)
    config = records(root, "bucket_trace_records", "configured_*.json")
    warm = records(root, "bucket_trace_records", "warmup_completed_*.json")
    assert any(r["enabled"] and len(r["points"]) == 75 and max(r["points"]) == 2048 for r in config)
    assert any(r["complete"] and len(r["calls"]) == 75 and max(c["size"] for c in r["calls"]) == 2048 for r in warm)
    extended = records(root, "extended_bucket_records", "extended_*.json")
    assert extended and all(r["changed_constants"] == [[2048, 8192]] and len(r["extended_points"]) == 99 for r in extended)
    producers = records(root, "decode_coherent_records", "producer_*.json")
    reducers = records(root, "decode_coherent_records", "reducer_*.json")
    assert producers and reducers
    for record in producers:
        selected = record["batch"] in (8, 16, 32, 40, 48, 56, 64)
        tile = 16 if selected else 32
        warps = 1 if selected else 2 if 8 <= record["batch"] <= 64 else 4
        assert record["configuration"] == dict(TILE_SIZE=tile, num_warps=warps, num_stages=1)
        assert record["compiled_metadata"]["num_warps"] == warps
        expected_segments = 8 if record["batch"] == 64 else 64 if record["batch"] == 1 else 4 if 16 <= record["batch"] <= 64 else 16
        assert record["segments"] == expected_segments
        assert record["generated_source_sha256"] == CV_SHA
        assert record["cache_service_mode"] == "candidate" and record["actual_cache_modifier"] == ".cv"
        assert digest(artifact(record["zero_initialization"]["generated_source"])) == CV_SHA
    assert {(r["batch"], r["segments"]) for r in producers} == {(r["batch"], r["segments"]) for r in reducers}
    assert all(r["actual_tile"] == (16 if r["batch"] in (8, 16, 32, 40, 48, 56, 64) else 32) for r in reducers)
    if official:
        assert any(r["batch"] == 64 and r["segments"] == 8 for r in producers)
    policy = records(root, "segment_policy_records", "choose_*.json")
    assert policy and all(r["selected_segments"] == (8 if r["batch"] == 64 else r["baseline_segments"]) for r in policy)
    gather = records(root, "bucket_trace_records", "gather_*.json")
    assert gather and any(r["index_dtype"] == "torch.int32" for r in gather)
    sql = records(root, "sql_reflection_records", "route_*.json")
    assert sql and all(r["reflection_scope"] == "one_requested_table" and not r["configuration_values_modified"] for r in sql)
    top = records(root, "topp_launch_records", "route_*.json")
    pure = [r for r in top if r["topp"] and not r["topk"]]
    assert pure
    for r in pure:
        assert r["source_sha256"] == "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"
        assert r["configuration"] == dict(num_warps=16, num_stages=1)
        assert r["compiled_metadata"]["num_warps"] == 16
    assert not records(root, "sampler_records", "sampler_*.json")
    if official:
        assert any(r["batch"] == 64 for r in pure)
    native = records(root, "native_prefill_records", "native_prefill_*.json")
    assert native and all(r["generated_kernel_sha256"] == "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e" for r in native)
    capacities = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", (root / "server.log").read_text())
    assert capacities and int(capacities[-1].replace(",", "")) >= 327680
    for name in ("before_metrics.txt", "after_metrics.txt"):
        values = re.findall(r"^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$", (root / name).read_text(), re.MULTILINE)
        assert values and all(float(value) == 0 for value in values)
    assert "No running processes found" in (root / "hardware_after.txt").read_text()
    observed = read(root / "observed_server_environment.json")
    assert observed["environment"] == state["profile_environment"] and len(observed["environment"]) == 35
    assert observed["command"] == state["server_command"]
    assert state["server_command"] == official_command()
    return dict(complete=True, warm_points=75, budget=2048, max_num_seqs=256, async_scheduling=True,
        CV_KV16_S8_verified=True, native_TopP_source="64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8", original_sampler=True,
        stream_interval=1, actual_environment_replayed=True, zero_preemptions=True,
        B64_binding_scope="Official/service C64 validated separately; Math requests use C8.")


def audit_math(root, state, profile, write=True):
    assert state["evalscope_exit_code"] == 0
    reports = list(root.glob("evalscope/*/reports/minicpm/math_500.json"))
    assert len(reports) == 1
    path = reports[0]
    report = read(path)
    execution = report["execution_summary"]
    assert execution["requested"] == execution["succeeded"] == report["num"] == 105
    assert execution["errored"] == 0 and execution["incomplete"] is False
    metric = next(m for m in report["metrics"] if m["identity"]["name"] == "accuracy")
    assert metric["num"] == 105
    run = path.parents[2]
    review_path = run / "reviews/minicpm/math_500_Level 3.jsonl"
    prediction_path = run / "predictions/minicpm/math_500_Level 3.jsonl"
    reviews = [json.loads(line) for line in review_path.read_text().splitlines()]
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    assert len(reviews) == len(predictions) == 105
    assert len({r["index"] for r in reviews}) == len({p["index"] for p in predictions}) == 105
    assert {r["index"] for r in reviews} == {p["index"] for p in predictions}
    assert all(p["model_output"]["error"] is None for p in predictions)
    scores = [r["sample_score"]["score"] for r in reviews]
    assert all(s["status"] == "success" for s in scores)
    values = [s["value"]["accuracy"] for s in scores]
    assert all(v in (0, 1) for v in values)
    correct = int(sum(values))
    assert abs(metric["score"] - correct / 105) < .000051
    result = dict(complete=True, fresh_math_for_original_command_profile=True, reused_original_accuracy=False,
        correct=correct, total=105, accuracy=correct / 105, official_requirement_passed=correct / 105 > .95,
        execution_summary=execution, incorrect_indices=[r["index"] for r,v in zip(reviews, values) if not v],
        generation_stop_reasons=dict(Counter(p["model_output"]["choices"][0]["stop_reason"] for p in predictions)),
        max_tokens_indices=[p["index"] for p in predictions if p["model_output"]["choices"][0]["stop_reason"] == "max_tokens"],
        profile=profile, server_command=state["server_command"], profile_environment=state["profile_environment"],
        frozen_sources=state["frozen_sources"], matched_screen_root=state["screen_root"],
        evaluation_command=state["evaluation_command"],
        raw_files={str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in (path, review_path, prediction_path)})
    if write:
        (root / "accuracy_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def csv_rows(root, pattern):
    paths = list((root / "benchmark_results").glob(pattern))
    assert len(paths) <= 1
    if not paths:
        return []
    with paths[0].open(newline="") as stream:
        return list(csv.DictReader(stream))


def normalized_environment(environment, root):
    return {key: value.replace(str(root) + "/", "<OUTPUT>/") for key, value in environment.items()}


def audit_benchmark(root, state, profile, quality=None, write=True):
    assert state["benchmark_exit_code"] == 0 and state["completed_runs"] == 4
    official = state["stage"] == "official"
    count = 256 if official else 64
    if official:
        assert quality["complete"] and quality["fresh_math_for_original_command_profile"]
        assert quality["total"] == 105 and quality["accuracy"] > .95
        assert not quality["reused_original_accuracy"]
        assert state["server_command"] == quality["server_command"] == official_command()
        assert state["frozen_sources"] == quality["frozen_sources"]
        assert normalized_environment(state["profile_environment"], state["output_root"]) == normalized_environment(quality["profile_environment"], state["math_root"])
    rows = csv_rows(root, "raw_runs_*.csv")
    summary = csv_rows(root, "summary_*.csv")
    assert len(rows) == 4 and len(summary) == 1
    assert rows == state["raw_runs"] and summary[0] == state["official_summary"]
    for row in rows:
        assert row["Run Status"] == "SUCCESS" and float(row["Successful Requests"]) == count
        assert [int(row[k]) for k in ("Prefill", "Decode", "Conc", "Num Prompts")] == [4096, 1024, 64, count]
        assert float(row["Total Input Tokens"]) == 4096 * count and float(row["Total Output Tokens"]) == 1024 * count
    for key in ("Total tok/s", "Output tok/s", "Mean TTFT (ms)"):
        assert abs(float(summary[0][key]) - statistics.mean(float(row[key]) for row in rows[1:])) < .011
    benchmark = root / "source_snapshot/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py"
    assert digest(benchmark) == state["official_script_sha256"] == BENCH_SHA
    total = float(summary[0]["Total tok/s"])
    ttft = float(summary[0]["Mean TTFT (ms)"])
    values = [float(row["Total tok/s"]) for row in rows[1:]]
    result = dict(complete=True, official_benchmark=official, original_serve_command_verified=True,
        total_tok_s=total, output_tok_s=float(summary[0]["Output tok/s"]), ttft_ms=ttft,
        requests_successful=count * 4, requests_per_run=count, raw_runs=rows, excluded_runs=[1],
        profile=profile, postfirst_spread_percent=100 * (max(values) - min(values)) / statistics.mean(values),
        adopted=False, full_16K_not_run=True, full_competition_eligibility_not_asserted=True,
        raw_response_scope="Original benchmark stdout and CSV; benchmark does not persist response bodies.")
    if official:
        result.update(fresh_math_for_original_command_profile=True, math_correct=quality["correct"],
            math_total=105, math_accuracy=quality["accuracy"], public_baseline_total_tok_s=2028.01,
            public_baseline_ttft_ms=11573.53, vs_public_throughput_percent=100 * (total / 2028.01 - 1),
            public_ttft_limit_ms=11573.53 * 1.01, passes_public_4k_performance_gates=total > 2028.01 * 1.01 and ttft <= 11573.53 * 1.01)
    if write:
        (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


