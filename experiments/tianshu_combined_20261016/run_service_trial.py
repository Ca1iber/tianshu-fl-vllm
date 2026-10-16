# SPDX-License-Identifier: Apache-2.0
"""Compare current profile with combined budget/maxseq/strictTopP/sampler/streaming candidate."""
import argparse
import hashlib
import importlib.util
import json
import re
import statistics
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "candidate"), required=True)
    parser.add_argument("--rounds", type=int, choices=(3,5), default=3)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    base = Path(__file__).resolve().parent
    foundation = base / "budget_trial.py"
    original_popen = subprocess.Popen
    actual_command = None
    actual_settings = None

    def popen(command, *positional, **kwargs):
        nonlocal actual_command, actual_settings
        if isinstance(command, list) and command[:2] == ["vllm", "serve"]:
            if args.mode == "candidate": command.extend(["--max-num-seqs", "64"])
            settings = dict(PYTHONPATH=str(base / "hooks"), TIANSHU_CACHE_SERVICE_MODE="candidate",
                            TIANSHU_MAX_SEQS_MODE=args.mode, TIANSHU_TOPP_PRECISION_SERVICE_MODE=args.mode,
                            TIANSHU_COMBO_MODE=args.mode,
                            TIANSHU_SAMPLER_FUSION_PROBE="1" if args.mode == "candidate" else "0",
                            TIANSHU_SAMPLER_RECORD_DIR=str(root / "sampler_records"),
                            TIANSHU_MAX_SEQS_RECORD_DIR=str(root / "max_seqs_records"))
            kwargs["env"] = dict(kwargs["env"], **settings)
            actual_command, actual_settings = list(command), settings
        return original_popen(command, *positional, **kwargs)

    spec = importlib.util.spec_from_file_location("_cache_profile_trial", foundation)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sampling_sources = {}
    for name, digest in (("topk_topp_triton.py", "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"),
                         ("topk_topp_sampler.py", "35b9a1ffeb1e50b73b0ddd1ffc0f968910ee21cf9cc5455dd879fde547b6ddc1")):
        path = Path("/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops") / name
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        sampling_sources[name] = (path, raw, digest)
    old_argv = sys.argv
    sys.argv = [str(foundation), "--output", str(root), "--mode", args.mode, "--rounds", str(args.rounds)]
    subprocess.Popen = popen
    try:
        code = module.main()
    finally:
        sys.argv = old_argv
        subprocess.Popen = original_popen
    state = json.loads((root / "result.json").read_text())
    state.update(combo_mode=args.mode, max_seqs_mode=args.mode, cache_mode="candidate", inherited_budget_mode=args.mode,
                 actual_service_environment=actual_settings, actual_service_command=actual_command,
                 active_hook=dict(path=str(base / "hooks/sitecustomize.py"),
                                  sha256=hashlib.sha256((base / "hooks/sitecustomize.py").read_bytes()).hexdigest()),
                 source_foundation_trial=str(foundation), source_foundation_sha256=hashlib.sha256(foundation.read_bytes()).hexdigest())
    if code == 0:
        snapshots = {}
        for name, (path, raw, digest) in sampling_sources.items():
            assert path.read_bytes() == raw
            dest = root / "source_snapshot" / ("native_" + name)
            dest.write_bytes(raw)
            snapshots[name] = dict(path=str(path), sha256=digest, snapshot="source_snapshot/native_"+name)
        state["installed_sampling_sources"] = snapshots
        assert state["phase"] == "complete"
        assert state["actual_budget"] == (4096 if args.mode == "control" else 16384)
        assert state["actual_bucket_points"] == (83 if args.mode == "control" else 131)
        assert state["stream_interval"] == (1 if args.mode == "control" else 4)
        assert len(state["short_4k_rounds"]) == args.rounds
        assert state["request_sampling_config"]["temperature"] == 1.0
        assert state["request_sampling_config"]["top_p"] == .95
        assert actual_command == state["command"]
        producers = state["decode_coherent_records"]
        assert producers and all(record["cache_service_mode"] == "candidate" for record in producers)
        expected_sha = "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
        assert all(record["generated_source_sha256"] == expected_sha for record in producers)
        assert all(record["actual_cache_modifier"] == ".cv" for record in producers)
        state.update(actual_cache_mode_verified=True, cache_modifier=".cv",
                     actual_decode_generated_sha256=expected_sha, max_num_seqs_only=False, combined_budget_maxseq_TopP_sampler_streaming=True,
                     sampling_native_topp_and_prefill_unchanged=False, sampling_parameters_and_RNG_unchanged=True, native_prefill_unchanged=True)
    if code == 0:
        records = [json.loads(path.read_text()) for path in (root / "max_seqs_records").glob("worker_*.json")]
        assert records and all(record["mode"] == args.mode for record in records)
        sizes = {record["max_num_seqs"] for record in records}
        assert len(sizes) == 1 and min(sizes) >= 64
        assert sizes == ({64} if args.mode == "candidate" else {256})
        assert all(r["async_scheduling"] is True for r in records)
        assert all(record["max_num_batched_tokens"] == (4096 if args.mode == "control" else 16384) and record["model_max_len"] == 131072 for record in records)
        assert all(record["worker_source_sha256"] == state["installed_sources"]["worker.py"]["sha256"] for record in records)
        state.update(actual_worker_config_records=records, actual_max_num_seqs=sizes.pop(), actual_max_num_seqs_verified=True)
    if code == 0:
        top = [json.loads(path.read_text()) for path in (root / "topp_launch_records").glob("route_*.json")]
        assert top and all(record["precision_service_mode"] == args.mode for record in top)
        pure = [r for r in top if r["topp"] and not r["topk"]]
        assert pure and any(r["batch"] == 64 for r in pure)
        for record in pure:
            active = args.mode == "candidate"
            assert record["corrected_source_active"] == active
            assert record["actual_BLOCK_SIZE"] == (16384 if active else 8192)
            assert record["actual_BLOCK_SIZE_TRUNC"] == (8192 if active else 4096)
            assert record["configuration"] == dict(num_warps=16, num_stages=1)
            assert record["actual_source_sha256"] == ("029d805b1a19be91f74afa03afb25c834409c8f08b3d30f7eabd1771c4aefedf" if active else "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8")
        assert all(not r["corrected_source_active"] for r in top if r["topk"])
        state.update(actual_TopP_precision_records=top, actual_TopP_precision_mode_verified=True)
    if code == 0:
        samples = [json.loads(p.read_text()) for p in (root / "sampler_records").glob("sampler_*.json")]
        if args.mode == "candidate":
            assert samples and any(r["batch"] == 64 for r in samples)
            assert all(r["rng_generation_unchanged"] and r["top_k_top_p_unchanged"] for r in samples)
            assert all(r["kernel_sha256"] == "83739846d48d6e12710d9cdbb0da61debb733241e8aa45e1ea1317bdd2950e93" for r in samples)
        else:
            assert not samples
        capacities = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", (root / "server.log").read_text())
        assert capacities
        capacity = int(capacities[-1].replace(",", ""))
        assert capacity >= 327680
        metrics = sorted(root.glob("round*_metrics.txt"))
        assert len(metrics) == args.rounds * 2
        for p in metrics:
            values = re.findall(r"^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$", p.read_text(), re.MULTILINE)
            assert values and all(float(x) == 0 for x in values)
        excluded = [1] if args.rounds == 3 else [1,2]
        scored = [r["total_tokens_per_second"] for r in state["short_4k_rounds"][len(excluded):]]
        state["workload"]["excluded_rounds"] = excluded
        state["measured_round_stability"] = dict(mean=statistics.mean(scored), count=len(scored), excluded_rounds=excluded,
            min=min(scored), max=max(scored), population_stddev=statistics.pstdev(scored))
        state.pop("later_four_stability", None)
        state.update(actual_sampler_records=samples, actual_sampler_mode_verified=True,
            actual_KV_cache_tokens=capacity, all_round_preemptions_zero=True,
            new_math_required_before_adoption=True, bitwise_for_every_float_input_claimed=False)
    (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
