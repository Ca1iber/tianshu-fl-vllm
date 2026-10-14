# SPDX-License-Identifier: Apache-2.0
"""Audit a candidate-only feasibility screen and its explicitly reused reference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    state = json.loads((root / "result.json").read_text())
    proof = json.loads((root / "comparison.json").read_text())
    assert state["phase"] == "complete" and proof["complete"]
    assert proof["preliminary"] and proof["reference_reused"] and not proof["fresh_paired_control"]
    assert not proof["adopted"] and not proof["official_benchmark"] and proof["excluded_rounds"] == [1]
    assert len(state["frozen_sources"]) == 20
    for source, digest in state["frozen_sources"].items():
        rel = Path(source).relative_to("/workspace/vllm-plugin-FL/experiments")
        assert sha(repo / "experiments" / rel) == digest
        assert sha(root / "source_snapshot" / rel) == digest
    trials = []
    rounds = []
    for name in ("reference_control", "candidate"):
        folder = root / name
        trial = json.loads((folder / "result.json").read_text())
        trials.append(trial)
        assert trial["phase"] == "complete" and trial["actual_cache_mode_verified"] and trial["actual_graph_mode_verified"]
        assert trial["actual_bucket_points"] == 83 and trial["actual_budget"] == 4096
        assert trial["common_gpu_memory_utilization"] == .70 and trial["actual_KV_cache_tokens"] >= 327680
        command = trial["command"]
        assert command[command.index("--gpu-memory-utilization") + 1] == "0.70"
        assert command[command.index("--max-model-len") + 1] == "131072"
        assert trial["request_sampling_config"] == dict(temperature=1., top_p=.95, repetition_penalty=1.,
                                                       logprobs=None, ignore_eos=True, stream=True)
        for file, metadata in trial["installed_sources"].items():
            assert sha(folder / "source_snapshot" / file) == metadata["sha256"]
        assert len(trial["short_4k_rounds"]) == 3
        for row in trial["short_4k_rounds"]:
            assert row["successful_requests"] == len(row["responses"]) == 64
            assert len({r["seed"] for r in row["responses"]}) == 64
            for r in row["responses"]:
                assert r["usage"] == dict(prompt_tokens=4096, completion_tokens=1024, total_tokens=5120)
                assert r["text"] and isinstance(r["text"], str)
                assert 0 < r["ttft_seconds"] <= r["wall_seconds"] <= row["duration_seconds"]
            assert abs(row["total_tokens_per_second"] - 327680 / row["duration_seconds"]) < 1e-8
            assert abs(row["output_tokens_per_second"] - 65536 / row["duration_seconds"]) < 1e-8
            assert abs(row["mean_ttft_seconds"] - statistics.mean(r["ttft_seconds"] for r in row["responses"])) < 1e-9
            rounds.append(dict(group=name, round=row["round"], total_tok_s=row["total_tokens_per_second"],
                               ttft_ms=row["mean_ttft_seconds"]*1000))
        metrics = list(folder.glob("round*_metrics.txt"))
        assert len(metrics) == 6
        for path in metrics:
            samples = re.findall(r"^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$", path.read_text(), re.MULTILINE)
            assert samples and all(float(x) == 0 for x in samples)
        warmed = [json.loads(p.read_text()) for p in (folder / "bucket_trace_records").glob("warmup_completed_*.json")]
        assert any(r["complete"] and len(r["calls"]) == 83 and max(c["size"] for c in r["calls"]) == 4096 for r in warmed)
        workers = [json.loads(p.read_text()) for p in (folder / "piecewise_records").glob("worker_*.json")]
        assert workers == trial["actual_worker_config_records"]
        count = 18 if name == "candidate" else 83
        mode = "FULL_AND_PIECEWISE" if name == "candidate" else "FULL_DECODE_ONLY"
        assert workers and all(len(r["capture_points"]) == count and r["actual_cudagraph_mode"] == mode for r in workers)
        assert all(r["max_num_seqs"] == 256 and r["max_num_batched_tokens"] == 4096 for r in workers)
        producers = [json.loads(p.read_text()) for p in (folder / "decode_coherent_records").glob("producer_*.json")]
        reducers = [json.loads(p.read_text()) for p in (folder / "decode_coherent_records").glob("reducer_*.json")]
        assert producers and reducers
        for record in producers:
            selected = record["batch"] in (8,16,32,40,48,56,64)
            tile = 16 if selected else 32
            warps = 1 if selected else 2 if 8 <= record["batch"] <= 64 else 4
            assert record["configuration"] == dict(TILE_SIZE=tile,num_warps=warps,num_stages=1)
            assert record["compiled_metadata"]["num_warps"] == warps
            assert record["actual_cache_modifier"] == ".cv" and record["cache_service_mode"] == "candidate"
            assert record["generated_source_sha256"] == "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
            generated = Path(record["zero_initialization"]["generated_source"])
            remote_folder = Path(trial["actual_composed_environment"]["TIANSHU_DECODE_COHERENT_RECORD_DIR"]).parent
            assert sha(folder / generated.relative_to(remote_folder)) == record["generated_source_sha256"]
        assert {(r["batch"],r["segments"]) for r in producers} == {(r["batch"],r["segments"]) for r in reducers}
        assert all(r["actual_tile"] == (16 if r["batch"] in (8,16,32,40,48,56,64) else 32) for r in reducers)
        assert any(r["batch"] == 64 and r["segments"] == 8 for r in producers)
        top = trial["topp_launch_records"]
        assert top and all(r["configuration"] == dict(num_warps=16, num_stages=1) for r in top)
        assert "No running processes found" in (folder / "hardware_after.txt").read_text()
    ref, candidate = trials
    assert ref["installed_sources"] == candidate["installed_sources"] and ref["workload"] == candidate["workload"]
    command_a, command_b = list(ref["command"]), list(candidate["command"])
    index = command_b.index("--compilation-config") + 1
    assert json.loads(command_a[index]) == dict(cudagraph_mode="FULL_DECODE_ONLY")
    assert json.loads(command_b[index]) == dict(cudagraph_mode="FULL_AND_PIECEWISE")
    command_b[index] = command_a[index]
    assert command_a == command_b
    folder = root / "candidate"
    policies = [json.loads(p.read_text()) for p in (folder / "piecewise_records").glob("capture_policy_*.json")]
    aliases = [json.loads(p.read_text()) for p in (folder / "piecewise_records").glob("alias_*.json")]
    assert policies and aliases
    assert all(len(p["actual_capture_points"]) == 18 and len(p["warm_points"]) == 83 and p["original83_warm_points_preserved"] for p in policies)
    for record in aliases:
        assert record["actual_shared_pointer"] and record["scope"] == "FL GraphWrapper graph-pool output only"
        assert record["graph_source_sha256"] == "ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2"
        assert record["extension"]["cpp_source_sha256"] == sha(repo / "experiments/tianshu_weak_alias_20261014/weak_alias.cpp")
        assert record["extension"]["shared_object_sha256"] == "7a074e25205cc52cb3a2b0e532e6812d53e2c4142b967182bb8ed55012a41040"
    scores = [statistics.mean(r["total_tokens_per_second"] for r in t["short_4k_rounds"][1:]) for t in trials]
    assert scores == [proof["reference_total_tok_s"], proof["candidate_total_tok_s"]]
    gain = 100*(scores[1]/scores[0]-1)
    assert abs(gain-proof["gain_percent"]) < 1e-9
    audit = dict(complete=True, preliminary=True, reference_reused=True, fresh_paired_control=False,
                 source20_verified=True, reference192_and_candidate192_requests_recomputed=True,
                 candidate_actual18_capture_and83_GEMM_warm_points_verified=True,
                 actual_alias_extension_and_zero_preemptions_verified=True,
                 reference_total_tok_s=scores[0], candidate_total_tok_s=scores[1], gain_percent=gain, rounds=rounds)
    (root / "local_download_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit))


if __name__ == "__main__": main()
