# SPDX-License-Identifier: Apache-2.0
"""Audit max_num_seqs64 A/B with raw requests, actual worker config and kernel provenance."""
import argparse
import hashlib
import json
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
    assert proof["excluded_rounds"] == [1,2]
    assert proof["frozen_sources"] == state["frozen_sources"] and len(state["frozen_sources"]) == 19
    for path, digest in state["frozen_sources"].items():
        relative = Path(path).relative_to("/workspace/vllm-plugin-FL")
        assert sha(repo / relative) == digest
        assert sha(root / "source_snapshot" / relative.relative_to("experiments")) == digest
    trials, rounds_audit = [], []
    for mode in ("control", "candidate"):
        folder = root / mode
        trial = json.loads((folder / "result.json").read_text())
        trials.append(trial)
        assert trial["phase"] == "complete" and trial["max_seqs_mode"] == mode and trial["cache_mode"] == "candidate"
        assert trial["actual_cache_mode_verified"] and trial["actual_max_num_seqs_verified"]
        assert trial["actual_budget"] == 4096 and trial["actual_bucket_points"] == 83
        assert trial["actual_threshold"] == 0
        assert trial["request_sampling_config"] == proof["request_sampling_config"]
        assert trial["request_sampling_config"] == dict(temperature=1., top_p=.95,
            repetition_penalty=1., logprobs=None, ignore_eos=True, stream=True)
        assert trial["actual_service_command"] == trial["command"]
        assert sha(repo / Path(trial["active_hook"]["path"]).relative_to("/workspace/vllm-plugin-FL")) == trial["active_hook"]["sha256"]
        for name, metadata in trial["installed_sources"].items():
            assert sha(folder / "source_snapshot" / name) == metadata["sha256"]
        rows = trial["short_4k_rounds"]
        assert len(rows) == 5
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
        assert abs(statistics.mean(row["total_tokens_per_second"] for row in rows[2:]) - proof[mode + "_total_tok_s"]) < 1e-8
        expected_sha = "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
        assert trial["actual_decode_generated_sha256"] == expected_sha
        assert trial["cache_modifier"] == ".cv"
        worker_records = [json.loads(p.read_text()) for p in (folder / "max_seqs_records").glob("worker_*.json")]
        assert sorted(worker_records,key=lambda r:r["pid"]) == sorted(trial["actual_worker_config_records"],key=lambda r:r["pid"])
        assert worker_records
        for record in worker_records:
            assert record["mode"] == mode and record["worker_class"] == "WorkerFL"
            assert record["max_num_seqs"] == trial["actual_max_num_seqs"] >= 64
            if mode == "candidate": assert record["max_num_seqs"] == 64
            assert record["max_num_batched_tokens"] == 4096 and record["model_max_len"] == 131072
            assert record["worker_source_sha256"] == trial["installed_sources"]["worker.py"]["sha256"]
            assert record["hook_source_sha256"] == sha(repo / "experiments/tianshu_max_seqs_20261014/hooks/sitecustomize.py")
        top = [json.loads(p.read_text()) for p in (folder / "topp_launch_records").glob("route_*.json")]
        assert top == trial["topp_launch_records"] or sorted(top,key=lambda r:(r["batch"],r["topk"],r["topp"])) == sorted(trial["topp_launch_records"],key=lambda r:(r["batch"],r["topk"],r["topp"]))
        assert trial["actual_TopP_precision_mode_verified"]
        pure = [r for r in top if r["topp"] and not r["topk"]]
        assert pure and any(r["batch"] == 64 for r in pure)
        for r in top:
            active = mode == "candidate" and r["topp"] and not r["topk"]
            assert r["precision_service_mode"] == mode and r["corrected_source_active"] == active
            assert r["actual_BLOCK_SIZE"] == (16384 if active else 8192)
            assert r["actual_BLOCK_SIZE_TRUNC"] == (8192 if active else 4096)
            assert r["configuration"] == dict(num_warps=16,num_stages=1)
            assert r["compiled_metadata"]["num_warps"] == 16 and r["compiled_metadata"]["num_stages"] == 1
            assert r["actual_source_sha256"] == ("029d805b1a19be91f74afa03afb25c834409c8f08b3d30f7eabd1771c4aefedf" if active else "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8")
            assert r["hook_source_sha256"] == sha(repo / "experiments/tianshu_topp_precision_service_20261014/hooks/top_p_sitecustomize.py")
            if active:
                generated = folder / "topp_launch_records" / Path(r["generation"]["generated_source"]).parent.name / "generated_topp_precision.py"
                assert sha(generated) == r["actual_source_sha256"]
                assert r["generation"]["native_TopK_branch_AST_unchanged"]
                assert r["generation"]["reversible_to_exact_native_source"]
        configured = [json.loads(p.read_text()) for p in (folder / "bucket_trace_records").glob("configured_*.json")]
        warmed = [json.loads(p.read_text()) for p in (folder / "bucket_trace_records").glob("warmup_completed_*.json")]
        assert any(r["enabled"] and len(r["points"]) == 83 and max(r["points"]) == 4096 for r in configured)
        assert any(r["complete"] and len(r["calls"]) == 83 and max(c["size"] for c in r["calls"]) == 4096 for r in warmed)
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
        assert trial["sql_reflection_enabled"] and trial["actual_sql_mode_verified"] and trial["actual_extended_policy_verified"]
        assert "No running processes found" in (folder / "hardware_after.txt").read_text()
    aa, bb = trials
    for key in ("installed_sources", "workload", "request_sampling_config"):
        assert aa[key] == bb[key]
    command_control, command_candidate = list(aa["command"]), list(bb["command"])
    assert "--max-num-seqs" not in command_control
    index = command_candidate.index("--max-num-seqs")
    assert command_candidate[index+1] == "64"
    del command_candidate[index:index+2]
    assert command_control == command_candidate
    assert proof["control_max_num_seqs"] == aa["actual_max_num_seqs"]
    assert proof["candidate_max_num_seqs"] == bb["actual_max_num_seqs"] == 64
    for x, y in zip(aa["short_4k_rounds"], bb["short_4k_rounds"]):
        assert [r["seed"] for r in x["responses"]] == [r["seed"] for r in y["responses"]]
    gain = 100 * (proof["candidate_total_tok_s"] / proof["control_total_tok_s"] - 1)
    assert abs(gain - proof["gain_percent"]) < 1e-9
    output = dict(complete=True, official_benchmark=False, adopted=False, source19_verified=True,
        all640_requests_checked=True, actual_83_warm_points_both=True,
        actual_worker_max_num_seqs_and_cache_provenance_verified=True,
        sampling_native_topp_and_prefill_unchanged=False, combined_TopP_precision_and_maxseq64=True, control_total_tok_s=proof["control_total_tok_s"],
        candidate_total_tok_s=proof["candidate_total_tok_s"], gain_percent=gain, rounds=rounds_audit,
        new_math_run=False)
    (root / "local_download_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
