# SPDX-License-Identifier: Apache-2.0
"""Native top-p launch sweep on actual MiniCPM vocabulary; no sampling changes."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import signal
import statistics
import subprocess

SOURCE_SHA = "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    deadline = datetime.fromisoformat(json.loads(Path(
        "/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json"
    ).read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 900
    signal.alarm(min(remaining, 1800))
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    root.mkdir(parents=True)
    import torch
    native = importlib.import_module("vllm.v1.sample.ops.topk_topp_triton")
    source = Path(native.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    config = json.loads(Path("/workspace/MiniCPM5-2B/config.json").read_text())
    vocab = config["vocab_size"]
    assert vocab == 130560
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    (snapshot / "native_topk_topp_triton.py").write_bytes(source)
    (snapshot / "probe_launch.py").write_bytes(Path(__file__).read_bytes())
    data = dict(complete=False, numerical_checks_passed=False, official_benchmark=False, adopted=False,
                native_source_sha256=SOURCE_SHA, vocab_size=vocab, body_grid_tiles_and_rng_unchanged=True,
                timing_includes_identical_input_restore=True, cases=[])
    kernel = native._topk_topp_kernel

    class Router:
        def __init__(self, warps, stages):
            self.warps, self.stages = warps, stages
            self.metadata = None

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                if self.warps is not None:
                    kwargs.update(num_warps=self.warps, num_stages=self.stages)
                compiled = kernel[grid](*args, **kwargs)
                self.metadata = {k: getattr(compiled.metadata, k, None)
                                 for k in ("num_warps", "num_stages", "shared", "num_ctas")}
                return compiled
            return launch

    def save():
        p = root / "result.tmp"
        p.write_text(json.dumps(data, indent=2) + "\n")
        p.replace(root / "result.json")

    plans = [("default", None, None), ("w2s1", 2, 1), ("w4s1", 4, 1),
             ("w8s1", 8, 1), ("w16s1", 16, 1), ("w2s3", 2, 3), ("w8s3", 8, 3)]
    cases = [(8, "normal", False), (32, "normal", False), (64, "normal", False),
             (64, "peaked", False), (8, "masked", False), (8, "ties", False),
             (32, "normal", True)]
    save()
    try:
        with torch.inference_mode():
            for ci, (batch, distribution, topk) in enumerate(cases):
                torch.manual_seed(2026101389 + ci)
                logits = torch.randn(batch, vocab, device="cuda", dtype=torch.float32)
                if distribution == "peaked":
                    logits.mul_(4)
                elif distribution == "masked":
                    logits[:, ::3] = -float("inf")
                elif distribution == "ties":
                    logits = logits.round()
                p = torch.full((batch,), .95, device="cuda", dtype=torch.float32)
                k = torch.full((batch,), 100, device="cuda", dtype=torch.int32) if topk else None
                native._topk_topp_kernel = kernel
                expected = native.apply_top_k_top_p_triton(logits.clone(), k, p)
                torch.cuda.synchronize()
                assert bool(torch.isfinite(expected).any(dim=1).all())
                # Convert onCPU after transfer: CoreX GPU FP64 conversion is not reliable.
                cpu_logits = logits.cpu().double()
                cpu_p = p.cpu().double()
                if topk:
                    threshold = cpu_logits.topk(100, dim=1).values[:, -1:]
                    cpu_logits = cpu_logits.masked_fill(cpu_logits < threshold, -float('inf'))
                cpu_probs = cpu_logits.softmax(dim=1)
                sorted_logits, sorted_ids = cpu_logits.sort(dim=1, descending=True, stable=True)
                sorted_probs = cpu_probs.gather(1, sorted_ids)
                sorted_keep = sorted_probs.cumsum(dim=1) - sorted_probs < cpu_p[:, None]
                reference_mask = torch.zeros_like(sorted_keep).scatter_(1, sorted_ids, sorted_keep)
                reference_mask &= torch.isfinite(cpu_logits)
                row = dict(batch=batch, distribution=distribution, topk=topk,
                           baseline_kept_counts=torch.isfinite(expected).sum(dim=1).cpu().tolist(), variants=[])
                graphs = []
                for name, warps, stages in plans:
                    entry = dict(label=name, requested_warps=warps, requested_stages=stages)
                    router = Router(warps, stages)
                    native._topk_topp_kernel = router
                    working = logits.clone()
                    try:
                        actual = native.apply_top_k_top_p_triton(working, k, p)
                        torch.cuda.synchronize()
                        exact = torch.equal(actual, expected)
                        actual_mask = torch.isfinite(actual).cpu()
                        kept_mass = (cpu_probs * actual_mask).sum(dim=1)
                        smallest = torch.where(actual_mask, cpu_probs, float('inf')).min(dim=1).values
                        minimum_kept = torch.where(actual_mask, cpu_logits, float('inf')).min(dim=1).values
                        maximum_removed = torch.where(~actual_mask, cpu_logits, -float('inf')).max(dim=1).values
                        nonempty = actual_mask.any(dim=1)
                        # Full probability-mass/minimality and ordering checks with1e-6 FP tolerance.
                        conformance = bool((nonempty & (kept_mass >= cpu_p - 1e-6)
                                            & (kept_mass - smallest <= cpu_p + 1e-6)
                                            & (maximum_removed <= minimum_kept)).all())
                        entry.update(cpu_reference_exact=bool(torch.equal(actual_mask, reference_mask)),
                                     cpu_reference_mask_difference=int((actual_mask != reference_mask).sum()),
                                     cpu_mass_min=float(kept_mass.min()), cpu_mass_max=float(kept_mass.max()),
                                     cpu_shortfall_max=float((cpu_p - kept_mass).max()),
                                     cpu_minimality_excess_max=float((kept_mass - smallest - cpu_p).max()),
                                     cpu_semantic_checks_passed=conformance)
                        mask_equal = torch.equal(torch.isfinite(actual), torch.isfinite(expected))
                        entry.update(exact_equal=exact, keep_mask_equal=mask_equal,
                                     mask_difference=int((torch.isfinite(actual) != torch.isfinite(expected)).sum().item()),
                                     compiled_metadata=router.metadata)
                        if exact or conformance:
                            replay_expected = actual.clone()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                working.copy_(logits)
                                native.apply_top_k_top_p_triton(working, k, p)
                            graph.replay()
                            torch.cuda.synchronize()
                            assert torch.equal(working, replay_expected)
                            graphs.append((entry, graph, working))
                    except Exception as error:
                        entry["error"] = repr(error)
                    row["variants"].append(entry)
                times = {entry["label"]: [] for entry, graph, working in graphs}
                for repeat in range(7):
                    order = graphs[repeat % len(graphs):] + graphs[:repeat % len(graphs)]
                    for entry, graph, working in order:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(10):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        times[entry["label"]].append(start.elapsed_time(end) / 10)
                for entry, graph, working in graphs:
                    entry.update(raw_ms=times[entry["label"]], median_ms=statistics.median(times[entry["label"]]))
                data["cases"].append(row)
                save()
                print(json.dumps(row), flush=True)
                del graphs, logits, expected, actual, working, p, k
                torch.cuda.empty_cache()
        passed = [name for name, warps, stages in plans
                  if all(next(v for v in c["variants"] if v["label"] == name).get("exact_equal", False)
                         and "error" not in next(v for v in c["variants"] if v["label"] == name)
                         for c in data["cases"])]
        semantic = [name for name, warps, stages in plans if all(
            next(v for v in c['variants'] if v['label'] == name).get('cpu_semantic_checks_passed', False)
            for c in data['cases'])]
        data.update(complete=True, numerical_checks_passed="default" in passed, exact_passing_plans=passed,
                    CPU_semantic_passing_plans=semantic, CPU_conversion_after_transfer=True,
                    candidate_service_not_run=True, independent_math_not_run=True)
        save()
    finally:
        native._topk_topp_kernel = kernel


if __name__ == "__main__":
    main()
