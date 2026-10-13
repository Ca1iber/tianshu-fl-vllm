# SPDX-License-Identifier: Apache-2.0
"""Guarded BF16 greedy fast path: exact indices and paired full-sampler cost."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    root.mkdir(parents=True)
    stop = datetime.fromisoformat(json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())["stop_at_utc"])
    signal.alarm(min(1200, int((stop - datetime.now(timezone.utc)).total_seconds())))
    import torch
    import vllm_fl
    import flag_gems
    from vllm.v1.sample.sampler import Sampler
    from implementation import sample_direct, supported
    torch.set_num_threads(8)
    base = Path(__file__).resolve().parent
    source = Path("/usr/local/lib/python3.12/site-packages/vllm/v1/sample/sampler.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "1a2f18ddc91c89f3744ab5a93624be056bf20b7a8b2e6e13d0236182718a2253"
    snapshots = root / "source_snapshot"
    snapshots.mkdir()
    frozen = {}
    for path in (Path(__file__), base / "implementation.py", source):
        raw = path.read_bytes()
        (snapshots / path.name).write_bytes(raw)
        frozen[str(path)] = hashlib.sha256(raw).hexdigest()
    data = dict(complete=False, official_benchmark=False, adopted=False, cases=[],
                frozen_sources=frozen, baseline="Original Sampler.forward inside explicit flag_gems.use_gems")

    def save():
        (root / "result.json").write_text(json.dumps(data, indent=2) + "\n")

    def metadata(batch):
        return SimpleNamespace(
            all_greedy=True, all_random=False, max_num_logprobs=None,
            logprob_token_ids=None, no_penalties=True, allowed_token_ids_mask=None,
            bad_words_token_ids={}, logitsprocs=SimpleNamespace(non_argmax_invariant=[], argmax_invariant=[]),
            thinking_budget_state_holder=None, output_token_ids=[[] for _ in range(batch)],
        )

    sampler = Sampler()
    mutations = dict(all_greedy=False, all_random=True, max_num_logprobs=0,
                     logprob_token_ids={0: [1]}, no_penalties=False,
                     allowed_token_ids_mask=True, bad_words_token_ids={0: [[1]]},
                     logitsprocs=SimpleNamespace(non_argmax_invariant=[object()]),
                     thinking_budget_state_holder=SimpleNamespace(has_tracked_requests=lambda: True))
    save()
    with flag_gems.use_gems():
        data["argmax_cuda_registration"] = [x for x in torch._C._dispatch_dump_table("aten::argmax").splitlines() if x.startswith("CUDA:")]
        assert any("flag_gems" in x for x in data["argmax_cuda_registration"])
        for batch, strided in ((1, False), (7, False), (64, False), (64, True)):
            for dtype in (torch.bfloat16, torch.float32):
                for pattern in ("normal", "zero_tie", "nan_first", "infinity", "negative_infinity", "large"):
                    torch.manual_seed(2026101400 + batch)
                    storage = torch.randn(batch, 130560 + (29 if strided else 0), device="cuda", dtype=dtype)
                    x = storage[:, :130560]
                    if pattern == "zero_tie":
                        x.zero_(); x[:, 1] = 5.; x[:, -1] = 5.
                    elif pattern == "nan_first":
                        x[:, 2] = float("nan"); x[:, -1] = float("nan")
                    elif pattern == "infinity":
                        x[:, 3] = float("inf"); x[:, -1] = float("inf")
                    elif pattern == "negative_infinity":
                        x.fill_(-float("inf"))
                    elif pattern == "large":
                        x.mul_(1e30)
                    meta = metadata(batch)
                    assert supported(x, meta)
                    if pattern == "normal":
                        for key, value in mutations.items():
                            bad = metadata(batch); setattr(bad, key, value)
                            assert not supported(x, bad), key
                        assert not supported(x.cpu(), meta)
                        assert not supported(x[:, :13], meta)
                    before = x.view(torch.int16 if dtype == torch.bfloat16 else torch.int32).clone()
                    expected = x.cpu().float().argmax(-1).to(torch.int32).unsqueeze(-1)
                    rng = torch.cuda.get_rng_state()
                    for _ in range(3):
                        output = sample_direct(x)
                        native = sampler(x, meta).sampled_token_ids
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = sample_direct(x)
                    native_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(native_graph):
                        native = sampler(x, meta).sampled_token_ids
                    graph.replay(); native_graph.replay(); torch.cuda.synchronize()
                    assert output.dtype == torch.int32 and output.shape == (batch, 1)
                    assert torch.equal(output.cpu(), expected)
                    assert torch.equal(before, x.view(before.dtype))
                    assert torch.equal(rng, torch.cuda.get_rng_state())
                    entry = dict(batch=batch, dtype=str(dtype), strided=strided, pattern=pattern,
                                 CPU_and_graph_exact=True, input_bits_unchanged=True, RNG_unchanged=True,
                                 native_exact=bool(torch.equal(native.cpu(), expected)))
                    if pattern != "nan_first":
                        assert entry["native_exact"]
                    if pattern == "normal":
                        times = [[], []]
                        for repeat in range(7):
                            for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(50):
                                    (native_graph if index == 0 else graph).replay()
                                end.record(); end.synchronize()
                                times[index].append(start.elapsed_time(end) / 50)
                        b, c = map(statistics.median, times)
                        entry.update(baseline_ms=b, candidate_ms=c, time_change_percent=100*(c/b-1), raw_ms=times)
                    data["cases"].append(entry); save()
                    print(json.dumps(entry), flush=True)
                    del output, native, graph, native_graph, x, storage, before
                    torch.cuda.empty_cache()
    data.update(complete=True, all_candidate_checks_passed=True,
                all_non_NaN_native_exact=True, guard_mutations_tested=list(mutations))
    save()


if __name__ == "__main__":
    main()
