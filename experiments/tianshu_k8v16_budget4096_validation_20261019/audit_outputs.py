# SPDX-License-Identifier: Apache-2.0
"""Original dataset/CSV verification with honest labels for the explicit budget4096 profile."""
from collections import Counter
import csv,json,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_original_command_20261016'))
from validation_profile import BENCH_SHA,digest,read,csv_rows,normalized_environment,official_command

def budget4096_command():
    return official_command()+['--max-num-batched-tokens','4096']

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
    result = dict(complete=True, fresh_math_for_budget4096_profile=True, reused_original_accuracy=False,
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

def audit_benchmark(root, state, profile, quality=None, write=True):
    assert state["benchmark_exit_code"] == 0 and state["completed_runs"] == 4
    official = state["stage"] == "official"
    count = 256 if official else 64
    if official:
        assert quality["complete"] and quality["fresh_math_for_budget4096_profile"]
        assert quality["total"] == 105 and quality["accuracy"] > .95
        assert not quality["reused_original_accuracy"]
        assert state["server_command"] == quality["server_command"] == budget4096_command()
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
    result = dict(complete=True, official_benchmark=official, original_serve_command_verified=False,explicit_budget4096_command_verified=True,
        total_tok_s=total, output_tok_s=float(summary[0]["Output tok/s"]), ttft_ms=ttft,
        requests_successful=count * 4, requests_per_run=count, raw_runs=rows, excluded_runs=[1],
        profile=profile, postfirst_spread_percent=100 * (max(values) - min(values)) / statistics.mean(values),
        adopted=False, full_16K_not_run=True, full_competition_eligibility_not_asserted=True,
        raw_response_scope="Original benchmark stdout and CSV; benchmark does not persist response bodies.")
    if official:
        result.update(fresh_math_for_budget4096_profile=True, math_correct=quality["correct"],
            math_total=105, math_accuracy=quality["accuracy"], public_baseline_total_tok_s=2028.01,
            public_baseline_ttft_ms=11573.53, vs_public_throughput_percent=100 * (total / 2028.01 - 1),
            public_ttft_limit_ms=11573.53 * 1.01, passes_public_4k_performance_gates=total > 2028.01 * 1.01 and ttft <= 11573.53 * 1.01)
    if write:
        (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
