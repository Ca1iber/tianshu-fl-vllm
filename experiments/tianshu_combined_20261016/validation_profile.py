# SPDX-License-Identifier: Apache-2.0
"""Source and runtime evidence for fresh validation of the combined profile."""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics


CV_SHA = "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
TOPP_SHA = "029d805b1a19be91f74afa03afb25c834409c8f08b3d30f7eabd1771c4aefedf"
SAMPLER_SHA = "83739846d48d6e12710d9cdbb0da61debb733241e8aa45e1ea1317bdd2950e93"
BENCH_SHA = "0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_check(sources):
    assert sources and all(digest(path) == sha for path, sha in sources.items())


def replay_environment(observed, old_root, new_root, inherited=None):
    """Replay the observed allowlist; do not inherit other experimental switches."""
    selected = observed["environment"]
    assert observed["frozen23_verified"] and len(selected) == 39
    assert selected["TIANSHU_COMBO_MODE"] == "candidate"
    assert selected["TIANSHU_BUCKET_RECHECK_TRACE"] == "1"
    assert selected["TIANSHU_TOPP_PRECISION_SERVICE_MODE"] == "candidate"
    assert selected["TIANSHU_SAMPLER_FUSION_PROBE"] == "1"
    assert selected["VLLM_FL_ILUVATAR_PREFILL_BUCKETS"] == "1"
    assert selected["VLLM_FL_ILUVATAR_LOGITS_GATHER"] == "1"
    original = str(Path(old_root))
    target = str(Path(new_root))
    mapped = {key: target + value[len(original):] if value.startswith(original + "/") else value
              for key, value in selected.items()}
    assert all(original not in value for value in mapped.values())
    environment = dict(os.environ if inherited is None else inherited)
    for key in list(environment):
        if key.startswith(("TIANSHU_", "VLLM_FL_", "FLAGGEMS_ILUVATAR_")) or key in {
            "VLLM_PLUGINS", "PYTHONPATH", "PYTHONUNBUFFERED", "TRITON_PRINT_AUTOTUNING", "CUDA_VISIBLE_DEVICES"
        }:
            del environment[key]
    environment.update(mapped)
    return environment, mapped


def prepare(root, screen_root, confirm_root):
    screen = read(screen_root / "comparison.json")
    proof = read(confirm_root / "comparison.json")
    assert read(confirm_root / "result.json")["phase"] == "complete"
    assert proof["complete"] and proof["rounds"] == 5 and proof["excluded_rounds"] == [1, 2]
    assert proof["successful_requests"] == 640 and proof["gain_percent"] >= 2
    assert min(proof["candidate_rounds"][2:]) > max(proof["control_rounds"][2:])
    assert Path(proof["screen_root"]) == screen_root
    assert proof["frozen_sources"] == screen["frozen_sources"]
    assert len(proof["frozen_sources"]) == 23
    frozen_check(proof["frozen_sources"])
    trial = read(confirm_root / "candidate/result.json")
    prior = read(screen_root / "candidate/result.json")
    assert trial["command"] == prior["command"]
    assert trial["actual_budget"] == 16384 and trial["actual_bucket_points"] == 131
    assert trial["actual_max_num_seqs"] == 64 and trial["stream_interval"] == 4
    observed = read(screen_root / "candidate/observed_server_environment.json")
    assert observed["command"] == prior["command"]
    environment, mapped = replay_environment(observed, screen_root / "candidate", root)
    installed = dict(trial["installed_sources"], **trial["installed_sampling_sources"])
    attention = Path("/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py")
    installed[attention.name] = dict(path=str(attention), sha256="bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0")
    sources = dict(proof["frozen_sources"])
    sources.update({metadata["path"]: metadata["sha256"] for metadata in installed.values()})
    for path in (Path(__file__), Path(__file__).with_name("run_validation.py")):
        sources[str(path)] = digest(path)
    benchmark = Path("/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py")
    assert digest(benchmark) == BENCH_SHA
    sources[str(benchmark)] = BENCH_SHA
    frozen_check(sources)
    for path, sha in sources.items():
        destination = root / "source_snapshot" / Path(path).relative_to("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(path).read_bytes())
        assert digest(destination) == sha
    for name, value in (("screen_comparison.json", screen), ("confirm_comparison.json", proof),
                        ("observed_screen_environment.json", observed)):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")
    return trial["command"], environment, mapped, sources


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
    assert workers and all(r["mode"] == "candidate" and r["max_num_seqs"] == 64 and
        r["async_scheduling"] is True and r["max_num_batched_tokens"] == 16384 and
        r["model_max_len"] == 131072 for r in workers)
    config = records(root, "bucket_trace_records", "configured_*.json")
    warm = records(root, "bucket_trace_records", "warmup_completed_*.json")
    assert any(r["enabled"] and len(r["points"]) == 131 and max(r["points"]) == 16384 for r in config)
    assert any(r["complete"] and len(r["calls"]) == 131 and max(c["size"] for c in r["calls"]) == 16384 for r in warm)
    extended = records(root, "extended_bucket_records", "extended_*.json")
    assert extended and all(r["changed_constants"] == [[2048, 16384]] and len(r["extended_points"]) == 131 for r in extended)
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
        assert r["precision_service_mode"] == "candidate" and r["corrected_source_active"]
        assert r["actual_source_sha256"] == TOPP_SHA
        assert r["actual_BLOCK_SIZE"] == 16384 and r["actual_BLOCK_SIZE_TRUNC"] == 8192
        assert r["configuration"] == dict(num_warps=16, num_stages=1)
        assert r["compiled_metadata"]["num_warps"] == 16
        generated = artifact(r["generation"]["generated_source"])
        assert digest(generated) == TOPP_SHA
        spec = importlib.util.spec_from_file_location("_validation_precision", Path(__file__).resolve().parents[1] / "tianshu_topp_precision_20261014/precision.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        native = root / "source_snapshot/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py"
        assert generated.read_bytes() == helper.render_source(native.read_bytes()).encode()
    samples = records(root, "sampler_records", "sampler_*.json")
    assert samples and all(r["rng_generation_unchanged"] and r["top_k_top_p_unchanged"] and r["kernel_sha256"] == SAMPLER_SHA for r in samples)
    if official:
        assert any(r["batch"] == 64 for r in pure) and any(r["batch"] == 64 for r in samples)
    native = records(root, "native_prefill_records", "native_prefill_*.json")
    assert native and all(r["generated_kernel_sha256"] == "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e" for r in native)
    capacities = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", (root / "server.log").read_text())
    assert capacities and int(capacities[-1].replace(",", "")) >= 327680
    for name in ("before_metrics.txt", "after_metrics.txt"):
        values = re.findall(r"^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$", (root / name).read_text(), re.MULTILINE)
        assert values and all(float(value) == 0 for value in values)
    assert "No running processes found" in (root / "hardware_after.txt").read_text()
    observed = read(root / "observed_server_environment.json")
    assert observed["environment"] == state["profile_environment"] and len(observed["environment"]) == 39
    assert observed["command"] == state["server_command"]
    return dict(complete=True, warm_points=131, budget=16384, max_num_seqs=64, async_scheduling=True,
        CV_KV16_S8_verified=True, strict_TopP_source=TOPP_SHA, sampler_source=SAMPLER_SHA,
        stream_interval=4, actual_environment_replayed=True, zero_preemptions=True,
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
    result = dict(complete=True, fresh_math_for_combined_profile=True, reused_original_accuracy=False,
        correct=correct, total=105, accuracy=correct / 105, official_requirement_passed=correct / 105 > .95,
        execution_summary=execution, incorrect_indices=[r["index"] for r,v in zip(reviews, values) if not v],
        generation_stop_reasons=dict(Counter(p["model_output"]["choices"][0]["stop_reason"] for p in predictions)),
        max_tokens_indices=[p["index"] for p in predictions if p["model_output"]["choices"][0]["stop_reason"] == "max_tokens"],
        profile=profile, server_command=state["server_command"], profile_environment=state["profile_environment"],
        frozen_sources=state["frozen_sources"], matched_confirm_root=state["confirm_root"],
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


def audit_official(root, state, quality, profile, write=True):
    assert state["benchmark_exit_code"] == 0 and state["completed_runs"] == 4
    assert quality["complete"] and quality["fresh_math_for_combined_profile"]
    assert quality["total"] == 105 and quality["accuracy"] > .95
    assert not quality["reused_original_accuracy"]
    assert state["server_command"] == quality["server_command"]
    assert state["frozen_sources"] == quality["frozen_sources"]
    assert normalized_environment(state["profile_environment"], state["output_root"]) == normalized_environment(quality["profile_environment"], state["math_root"])
    rows = csv_rows(root, "raw_runs_*.csv")
    summary = csv_rows(root, "summary_*.csv")
    assert len(rows) == 4 and len(summary) == 1
    assert rows == state["raw_runs"] and summary[0] == state["official_summary"]
    for row in rows:
        assert row["Run Status"] == "SUCCESS" and float(row["Successful Requests"]) == 256
        assert [int(row[k]) for k in ("Prefill", "Decode", "Conc", "Num Prompts")] == [4096, 1024, 64, 256]
        assert float(row["Total Input Tokens"]) == 1048576 and float(row["Total Output Tokens"]) == 262144
    for key in ("Total tok/s", "Output tok/s", "Mean TTFT (ms)"):
        assert abs(float(summary[0][key]) - statistics.mean(float(row[key]) for row in rows[1:])) < .011
    benchmark = root / "source_snapshot/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py"
    assert digest(benchmark) == state["official_script_sha256"] == BENCH_SHA
    total = float(summary[0]["Total tok/s"])
    ttft = float(summary[0]["Mean TTFT (ms)"])
    values = [float(row["Total tok/s"]) for row in rows[1:]]
    result = dict(complete=True, official_benchmark=True, fresh_math_for_combined_profile=True,
        total_tok_s=total, output_tok_s=float(summary[0]["Output tok/s"]), ttft_ms=ttft,
        requests_successful=1024, raw_runs=rows, excluded_runs=[1], math_correct=quality["correct"],
        math_total=105, math_accuracy=quality["accuracy"], previous_best_total_tok_s=4091.29,
        previous_best_ttft_ms=4281.52, vs_previous_best_throughput_percent=100 * (total / 4091.29 - 1),
        vs_previous_best_ttft_percent=100 * (ttft / 4281.52 - 1),
        vs_public_throughput_percent=100 * (total / 2028.01 - 1),
        postfirst_spread_percent=100 * (max(values) - min(values)) / statistics.mean(values),
        eligible_as_best=total > 4091.29, adopted=False, profile=profile, full_16K_not_run=True,
        raw_response_scope="Original benchmark stdout and CSV; benchmark does not persist response bodies.")
    if write:
        (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
