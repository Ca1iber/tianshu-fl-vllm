# SPDX-License-Identifier: Apache-2.0
"""Audit combined3/5-round profile, installed sampling sources, real component routes and complete responses."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import statistics
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    state = json.loads((root / "result.json").read_text())
    proof = json.loads((root / "comparison.json").read_text())
    assert state["phase"] == "complete" and proof["complete"]
    assert not proof["official_benchmark"] and not proof["adopted"] and not proof["control_reused"]
    rounds = proof["rounds"]
    assert rounds in (3,5)
    excluded = [1] if rounds == 3 else [1,2]
    assert proof["excluded_rounds"] == excluded == state["excluded_rounds"]
    assert proof["successful_requests"] == 2 * rounds * 64
    if rounds == 5:
        assert proof["screen_root"] is not None
    assert proof["frozen_sources"] == state["frozen_sources"] and len(state["frozen_sources"]) == 23
    for path, digest in state["frozen_sources"].items():
        relative = Path(path).relative_to("/workspace/vllm-plugin-FL")
        assert sha(repo / relative) == digest
        assert sha(root / "source_snapshot" / relative.relative_to("experiments")) == digest
    trials, rounds_audit = [], []
    for mode in ("control", "candidate"):
        folder = root / mode
        trial = json.loads((folder / "result.json").read_text())
        trials.append(trial)
        assert trial["phase"] == "complete" and trial["combo_mode"] == mode and trial["max_seqs_mode"] == mode and trial["cache_mode"] == "candidate"
        assert trial["actual_cache_mode_verified"] and trial["actual_max_num_seqs_verified"]
        budget = 4096 if mode == "control" else 16384
        points = 83 if mode == "control" else 131
        assert trial["actual_budget"] == budget and trial["actual_bucket_points"] == points
        assert trial["actual_KV_cache_tokens"] >= 327680 and trial["all_round_preemptions_zero"]
        capacities = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", (folder / "server.log").read_text())
        assert int(capacities[-1].replace(",", "")) == trial["actual_KV_cache_tokens"]
        metrics = sorted(folder.glob("round*_metrics.txt"))
        assert len(metrics) == rounds * 2
        for p in metrics:
            values = re.findall(r"^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$", p.read_text(), re.MULTILINE)
            assert values and all(float(x) == 0 for x in values)
        assert trial["actual_threshold"] == 0
        assert trial["stream_interval"] == (1 if mode == "control" else 4)
        assert trial["actual_TopP_precision_mode_verified"] and trial["actual_sampler_mode_verified"]
        assert trial["request_sampling_config"] == proof["request_sampling_config"]
        assert trial["request_sampling_config"] == dict(temperature=1., top_p=.95,
            repetition_penalty=1., logprobs=None, ignore_eos=True, stream=True)
        assert trial["actual_service_command"] == trial["command"]
        assert sha(repo / Path(trial["active_hook"]["path"]).relative_to("/workspace/vllm-plugin-FL")) == trial["active_hook"]["sha256"]
        for name, metadata in trial["installed_sources"].items():
            assert sha(folder / "source_snapshot" / name) == metadata["sha256"]
        rows = trial["short_4k_rounds"]
        assert len(rows) == rounds
        for i, row in enumerate(rows, 1):
            assert row["round"] == i and row["successful_requests"] == len(row["responses"]) == 64
            assert len({response["seed"] for response in row["responses"]}) == 64
            for response in row["responses"]:
                assert response["usage"] == dict(prompt_tokens=4096, completion_tokens=1024, total_tokens=5120)
                assert response["input_tokens"] == 4096 and response["output_tokens"] == 1024
                assert isinstance(response["text"], str) and response["text"]
                assert 0 < response["ttft_seconds"] <= response["wall_seconds"] <= row["duration_seconds"]
            assert abs(row["total_tokens_per_second"] - 64 * 5120 / row["duration_seconds"]) < 1e-8
            assert abs(row["output_tokens_per_second"] - 64 * 1024 / row["duration_seconds"]) < 1e-8
            assert abs(row["mean_ttft_seconds"] - statistics.mean(r["ttft_seconds"] for r in row["responses"])) < 1e-9
            rounds_audit.append(dict(mode=mode, round=i, total_tok_s=row["total_tokens_per_second"],
                output_tok_s=row["output_tokens_per_second"], ttft_ms=row["mean_ttft_seconds"] * 1000))
        assert [row["total_tokens_per_second"] for row in rows] == proof[mode + "_rounds"]
        assert abs(statistics.mean(row["total_tokens_per_second"] for row in rows[len(excluded):]) - proof[mode + "_total_tok_s"]) < 1e-8
        expected_sha = "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
        assert trial["actual_decode_generated_sha256"] == expected_sha
        assert trial["cache_modifier"] == ".cv"
        worker_records = [json.loads(p.read_text()) for p in (folder / "max_seqs_records").glob("worker_*.json")]
        assert sorted(worker_records,key=lambda r:r["pid"]) == sorted(trial["actual_worker_config_records"],key=lambda r:r["pid"])
        assert worker_records
        for record in worker_records:
            assert record["mode"] == mode and record["worker_class"] == "WorkerFL"
            assert record["max_num_seqs"] == trial["actual_max_num_seqs"] >= 64
            assert record["max_num_seqs"] == (256 if mode == "control" else 64)
            assert record["async_scheduling"] is True
            assert record["max_num_batched_tokens"] == budget and record["model_max_len"] == 131072
            assert record["worker_source_sha256"] == trial["installed_sources"]["worker.py"]["sha256"]
            assert record["hook_source_sha256"] == sha(repo / "experiments/tianshu_max_seqs_20261014/hooks/sitecustomize.py")
        configured = [json.loads(p.read_text()) for p in (folder / "bucket_trace_records").glob("configured_*.json")]
        warmed = [json.loads(p.read_text()) for p in (folder / "bucket_trace_records").glob("warmup_completed_*.json")]
        assert any(r["enabled"] and len(r["points"]) == points and max(r["points"]) == budget for r in configured)
        assert any(r["complete"] and len(r["calls"]) == points and max(c["size"] for c in r["calls"]) == budget for r in warmed)
        extensions = [json.loads(p.read_text()) for p in (folder / "extended_bucket_records").glob("extended_*.json")]
        assert extensions
        for record in extensions:
            assert record["helper_sha256"] == sha(repo / "experiments/tianshu_budget16384_20261016/extend_buckets.py")
            assert len(record["original_points"]) == 75 and len(record["extended_points"]) == 131
            assert record["extended_points"][-1] == 16384
            assert record["original_points"] == record["extended_points"][:75]
            assert record["changed_constants"] == [[2048,16384]]
        producers = [json.loads(p.read_text()) for p in (folder / "decode_coherent_records").glob("producer_*.json")]
        reducers = [json.loads(p.read_text()) for p in (folder / "decode_coherent_records").glob("reducer_*.json")]
        assert producers and reducers
        for record in producers:
            selected = record["batch"] in (8,16,32,40,48,56,64)
            tile = 16 if selected else 32
            warps = 1 if selected else 2 if 8 <= record["batch"] <= 64 else 4
            assert record["configuration"] == dict(TILE_SIZE=tile, num_warps=warps, num_stages=1)
            assert record["compiled_metadata"]["num_warps"] == warps
            assert record["generated_source_sha256"] == expected_sha
            assert record["cache_service_mode"] == "candidate"
            assert record["actual_cache_modifier"] == ".cv"
            assert record["cache_helper_sha256"] == sha(repo / "experiments/tianshu_decode_cache_policy_20261014/experimental_cache.py")
            cache = record["cache_policy_generation"]
            assert cache["generated_sha256"] == "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
            assert cache["original_L0_sha256"] == "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e"
            assert not cache["attention_arithmetic_changed"] and cache["only_3D_KV_load_cache_policy_changed"]
            generated = Path(record["zero_initialization"]["generated_source"])
            remote_folder = Path(trial["actual_composed_environment"]["TIANSHU_DECODE_COHERENT_RECORD_DIR"]).parent
            assert sha(folder / generated.relative_to(remote_folder)) == record["generated_source_sha256"]
        assert any(r["batch"] == 64 and r["segments"] == 8 and r["configuration"]["TILE_SIZE"] == 16 for r in producers)
        assert {(r["batch"], r["segments"]) for r in producers} == {(r["batch"], r["segments"]) for r in reducers}
        assert all(r["actual_tile"] == (16 if r["batch"] in (8,16,32,40,48,56,64) else 32) for r in reducers)
        sampling = trial["installed_sampling_sources"]
        for name, digest in (("topk_topp_triton.py", "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"),
                             ("topk_topp_sampler.py", "35b9a1ffeb1e50b73b0ddd1ffc0f968910ee21cf9cc5455dd879fde547b6ddc1")):
            assert sampling[name]["sha256"] == sha(folder / sampling[name]["snapshot"]) == digest
        top = [json.loads(p.read_text()) for p in (folder / "topp_launch_records").glob("route_*.json")]
        assert top == trial["actual_TopP_precision_records"] or sorted(top,key=lambda r:(r["batch"],r["topk"],r["topp"])) == sorted(trial["actual_TopP_precision_records"],key=lambda r:(r["batch"],r["topk"],r["topp"]))
        pure = [r for r in top if r["topp"] and not r["topk"]]
        assert pure and any(r["batch"] == 64 for r in pure)
        checked_generated = set()
        for record in top:
            assert record["precision_service_mode"] == mode
            active = mode == "candidate" and record["topp"] and not record["topk"]
            assert record["corrected_source_active"] == active
            assert record["actual_source_sha256"] == ("029d805b1a19be91f74afa03afb25c834409c8f08b3d30f7eabd1771c4aefedf" if active else "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8")
            assert record["actual_BLOCK_SIZE"] == (16384 if active else 8192)
            assert record["actual_BLOCK_SIZE_TRUNC"] == (8192 if active else 4096)
            assert record["configuration"] == dict(num_warps=16,num_stages=1)
            assert record["compiled_metadata"]["num_warps"] == 16
            if active:
                remote_output = Path(trial["actual_service_environment"]["TIANSHU_MAX_SEQS_RECORD_DIR"]).parent
                generated = folder / Path(record["generation"]["generated_source"]).relative_to(remote_output)
                assert sha(generated) == record["actual_source_sha256"]
                if generated not in checked_generated:
                    spec = importlib.util.spec_from_file_location("_offline_precision_generator",repo / "experiments/tianshu_topp_precision_20261014/precision.py")
                    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
                    assert generated.read_bytes() == helper.render_source((folder / sampling["topk_topp_triton.py"]["snapshot"]).read_bytes()).encode()
                    checked_generated.add(generated)
        samples = [json.loads(p.read_text()) for p in (folder / "sampler_records").glob("sampler_*.json")]
        assert sorted(samples,key=lambda x:(x["pid"],x["batch"])) == sorted(trial["actual_sampler_records"],key=lambda x:(x["pid"],x["batch"]))
        if mode == "candidate":
            assert samples and any(r["batch"] == 64 for r in samples)
            assert all(r["kernel_sha256"] == "83739846d48d6e12710d9cdbb0da61debb733241e8aa45e1ea1317bdd2950e93" for r in samples)
            assert all(r["rng_generation_unchanged"] for r in samples)
        else:
            assert not samples
        assert trial["sql_reflection_enabled"] and trial["actual_sql_mode_verified"] and trial["actual_extended_policy_verified"]
        assert "No running processes found" in (folder / "hardware_after.txt").read_text()
    aa, bb = trials
    for key in ("installed_sources", "workload", "request_sampling_config"):
        assert aa[key] == bb[key]
    command_control, command_candidate = list(aa["command"]), list(bb["command"])
    assert "--max-num-seqs" not in command_control and "--stream-interval" not in command_control
    for flag, expected in (("--max-num-seqs", "64"),("--stream-interval", "4")):
        index = command_candidate.index(flag)
        assert command_candidate[index+1] == expected
        del command_candidate[index:index+2]
    i = command_control.index("--max-num-batched-tokens") + 1
    assert command_control[i] == "4096" and command_candidate[i] == "16384"
    command_candidate[i] = "4096"
    assert command_control == command_candidate
    assert proof["control_max_num_seqs"] == aa["actual_max_num_seqs"] == 256
    assert proof["candidate_max_num_seqs"] == bb["actual_max_num_seqs"] == 64
    assert proof["control_budget"] == 4096 and proof["candidate_budget"] == 16384
    assert proof["warm_points"] == [83,131]
    for x, y in zip(aa["short_4k_rounds"], bb["short_4k_rounds"]):
        assert [r["seed"] for r in x["responses"]] == [r["seed"] for r in y["responses"]]
    gain = 100 * (proof["candidate_total_tok_s"] / proof["control_total_tok_s"] - 1)
    assert abs(gain - proof["gain_percent"]) < 1e-9
    output = dict(complete=True, official_benchmark=False, adopted=False, source23_verified=True,
        all_requests_checked=2 * rounds * 64, actual_83_and131_warm_points_verified=True,
        actual_worker_max_num_seqs_and_cache_provenance_verified=True,
        sampling_native_topp_and_prefill_unchanged=False, control_total_tok_s=proof["control_total_tok_s"],
        candidate_total_tok_s=proof["candidate_total_tok_s"], gain_percent=gain, rounds=rounds_audit,
        new_math_run=False, actual_sampling_sources_and_generated_TopP_bytes_verified=True,
        actual_sampler_fusion_B64_verified=True, actual_streaming_command_verified=True, zero_preemptions_both=True, capacity_both_at_least327680=True)
    (root / "local_download_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
