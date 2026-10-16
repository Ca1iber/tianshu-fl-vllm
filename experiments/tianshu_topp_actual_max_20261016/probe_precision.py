# SPDX-License-Identifier: Apache-2.0
"""Compare actual-max FP32 normalization without compensated reduction."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import signal
import statistics
import subprocess

from router import PLANS, Router

SOURCE_SHA = "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-at-utc", required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    deadline = datetime.fromisoformat(args.stop_at_utc)
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 900
    signal.alarm(min(remaining, 2100))
    hardware = subprocess.check_output(["ixsmi"], text=True)
    assert "No running processes found" in hardware
    root.mkdir(parents=True)
    (root / "hardware_before.txt").write_text(hardware)
    import torch
    torch.set_num_threads(8)
    native = importlib.import_module("vllm.v1.sample.ops.topk_topp_triton")
    source = Path(native.__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    assert json.loads(Path("/workspace/MiniCPM5-2B/config.json").read_text())["vocab_size"] == 130560
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for path in (Path(__file__), Path(__file__).with_name("router.py")):
        (snapshot / path.name).write_bytes(path.read_bytes())
    (snapshot / "native_topk_topp_triton.py").write_bytes(source.read_bytes())
    kernel = native._topk_topp_kernel
    from precision import load_kernel
    fixedkernel,precision_provenance=load_kernel(source,root/"precision_source")
    (snapshot/"precision.py").write_bytes(Path(__file__).with_name("precision.py").read_bytes())
    data = dict(complete=False, official_benchmark=False, adopted=False,
                native_source_sha256=SOURCE_SHA, vocab_size=130560, plans=PLANS, cases=[],
                baseline="adopted8192/4096/w16/s1", CPU_tolerance=1e-6,
                precision_generation=precision_provenance, corrected_candidates_require_strict_CPU=True,
                timing_includes_identical_input_restore=True, RNG_and_sampling_parameters_unchanged=True)

    def save():
        temp = root / "result.tmp"
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(root / "result.json")

    def oracle(logits, k, p):
        # Transfer first; CoreX device-to-FP64 conversion was unreliable in prior tests.
        values, probabilities = logits.cpu().double(), p.cpu().double()
        if k is not None:
            threshold = values.topk(100, dim=1).values[:, -1:]
            values = values.masked_fill(values < threshold, -float("inf"))
        probs = values.softmax(-1)
        ordered, indices = values.sort(dim=1, descending=True, stable=True)
        ordered_probs = probs.gather(1, indices)
        keep = ordered_probs.cumsum(1) - ordered_probs < probabilities[:, None]
        reference = torch.zeros_like(keep).scatter_(1, indices, keep) & torch.isfinite(values)
        return values, probs, probabilities, reference

    def assess(actual, expected, source_logits, reference, require_strict=False):
        values, probs, p_cpu, stable = reference
        mask = torch.isfinite(actual).cpu()
        mass = (probs * mask).sum(1)
        smallest = torch.where(mask, probs, float("inf")).min(1).values
        minimum_kept = torch.where(mask, values, float("inf")).min(1).values
        maximum_removed = torch.where(~mask, values, -float("inf")).max(1).values
        nonempty = mask.any(1)
        strict = bool((nonempty & (mass >= p_cpu - 1e-6) &
                      (mass - smallest <= p_cpu + 1e-6) & (maximum_removed <= minimum_kept)).all())
        finite = torch.isfinite(actual)
        retained_unchanged = bool(torch.equal(actual[finite], source_logits[finite]))
        exact = bool(torch.equal(actual, expected))
        return dict(exact_adopted_baseline=exact, CPU_strict_semantics=strict,
            relative_gate=retained_unchanged and (strict if require_strict else (exact or strict)),
            strict_required=require_strict,retained_values_unchanged=retained_unchanged,
            stable_CPU_mask_difference=int((mask != stable).sum()),
            adopted_mask_difference=int((torch.isfinite(actual) != torch.isfinite(expected)).sum()),
            kept_mass=mass.tolist(), smallest_kept_probability=smallest.tolist(),
            minimum_kept_logit=minimum_kept.tolist(), maximum_removed_logit=maximum_removed.tolist(),
            nonempty=nonempty.tolist(), CPU_p=p_cpu.tolist())

    cases = [(8, "normal", False), (32, "normal", False), (64, "normal", False),
             (64, "peaked", False), (8, "masked", False), (8, "ties", False), (32, "normal", True),(64,"bf16_normal",False),(64,"bf16_peaked",False),(8,"all_equal",False)]
    save()
    try:
        with torch.inference_mode():
            for index, (batch, distribution, topk) in enumerate(cases):
                torch.manual_seed(2026101421 + index)
                logits = torch.randn(batch, 130560, device="cuda", dtype=torch.float32)
                if distribution in ("peaked","bf16_peaked"): logits.mul_(4)
                elif distribution == "masked": logits[:, ::3] = -float("inf")
                elif distribution == "ties": logits = logits.round()
                elif distribution == "all_equal": logits.zero_()
                if distribution.startswith("bf16_"):logits=logits.to(torch.bfloat16).float()
                p = torch.full((batch,), .95, device="cuda", dtype=torch.float32)
                k = torch.full((batch,), 100, device="cuda", dtype=torch.int32) if topk else None
                native._topk_topp_kernel = Router(kernel, PLANS[0], corrected_kernel=fixedkernel)
                expected = native.apply_top_k_top_p_triton(logits.clone(), k, p)
                reference = oracle(logits, k, p)
                row = dict(batch=batch, distribution=distribution, topk=topk, variants=[])
                graphs = []
                for plan in PLANS:
                    router = Router(kernel, plan, root / "kernel_artifacts", corrected_kernel=fixedkernel)
                    native._topk_topp_kernel = router
                    working = logits.clone()
                    entry = dict(label=plan[0], requested_plan=plan)
                    try:
                        actual = native.apply_top_k_top_p_triton(working, k, p)
                        torch.cuda.synchronize()
                        entry.update(original=assess(actual, expected, logits, reference, require_strict=plan[0].startswith("fixed")), compiled_metadata=router.metadata)
                        if entry["original"]["relative_gate"]:
                            replay_expected = actual.clone()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                working.copy_(logits)
                                native.apply_top_k_top_p_triton(working, k, p)
                            graph.replay()
                            torch.cuda.synchronize()
                            assert torch.equal(working, replay_expected)
                            entry["original_graph_exact"] = True
                            graphs.append((entry, graph, working, plan))
                    except Exception as error:
                        if plan[0] == "adopted": raise
                        entry["error"] = repr(error)
                    row["variants"].append(entry)
                saved = logits.clone()
                logits.copy_(logits.flip((1,)))
                native._topk_topp_kernel = Router(kernel, PLANS[0], corrected_kernel=fixedkernel)
                changed_expected = native.apply_top_k_top_p_triton(logits.clone(), k, p)
                changed_reference = oracle(logits, k, p)
                timed = []
                for entry, graph, working, plan in graphs:
                    native._topk_topp_kernel = Router(kernel, plan, corrected_kernel=fixedkernel)
                    direct = native.apply_top_k_top_p_triton(logits.clone(), k, p)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(working, direct)
                    entry.update(changed=assess(direct, changed_expected, logits, changed_reference, require_strict=plan[0].startswith("fixed")), changed_graph_exact=True)
                    if entry["changed"]["relative_gate"]: timed.append((entry, graph, working, plan))
                logits.copy_(saved)
                times = {entry["label"]: [] for entry, _, _, _ in timed}
                for repetition in range(7):
                    order = timed[repetition % len(timed):] + timed[:repetition % len(timed)]
                    for entry, graph, _, _ in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(10): graph.replay()
                        end.record()
                        end.synchronize()
                        times[entry["label"]].append(start.elapsed_time(end) / 10)
                for entry, _, _, _ in timed:
                    entry.update(raw_ms=times[entry["label"]], median_ms=statistics.median(times[entry["label"]]))
                data["cases"].append(row)
                save()
                print(json.dumps(dict(case=index + 1, batch=batch, distribution=distribution,
                    variants=[dict(label=v["label"], ms=v.get("median_ms"), error=v.get("error"),
                                   gate=v.get("changed", {}).get("relative_gate")) for v in row["variants"]])), flush=True)
                del graphs, timed, logits, saved, working, actual, expected, direct, changed_expected, reference, changed_reference
                torch.cuda.empty_cache()
        valid = [plan[0] for plan in PLANS if all(any(v["label"] == plan[0] and "median_ms" in v
                    and v.get("original_graph_exact") and v.get("changed_graph_exact") and "error" not in v
                    for v in case["variants"]) for case in data["cases"])]
        assert "adopted" in valid and "identity" in valid
        data.update(complete=True, numerical_checks_passed=True, relative_passing_plans=valid,
                    independent_math_not_run=True, candidate_service_not_run=True)
    finally:
        native._topk_topp_kernel = kernel
        save()


if __name__ == "__main__": main()
