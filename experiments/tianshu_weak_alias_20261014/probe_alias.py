# SPDX-License-Identifier: Apache-2.0
"""Check alias ownership on CPU and replay with graph-pool managed CUDA allocations."""
import argparse
from datetime import datetime, timezone
import signal
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import weakref

from weak_alias import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 900
    signal.alarm(min(900, remaining))
    root = args.output.resolve()
    assert not root.exists()
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    root.mkdir(parents=True)
    import torch
    torch.set_num_threads(4)
    result = dict(complete=False, official_benchmark=False, adopted=False, CPU_cases=[], graph_cases=[])
    def save(): (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    save()
    extension, proof = load()
    result["extension"] = proof
    for shape, stride in [((4, 8), (8, 1)), ((4, 8), (10, 1)), ((4, 8), (1, 4))]:
        tensor = torch.empty_strided(shape, stride, dtype=torch.float32).uniform_()
        alias = extension.weak_alias(tensor)
        assert alias.data_ptr() == tensor.data_ptr() and alias.shape == tensor.shape and alias.stride() == tensor.stride()
        assert torch.equal(alias, tensor)
        owner = weakref.ref(tensor)
        del tensor
        gc.collect()
        assert owner() is None
        # Never dereference an alias after its backing CPU owner is destroyed.
        del alias
        result["CPU_cases"].append(dict(shape=shape, stride=stride, owner_released=True))
    torch.cuda.init()
    pool = torch.cuda.graph_pool_handle()
    live = []
    for shape in [(128, 256), (64, 256), (32, 256), (16, 256)]:
        inputs = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): torch.add(inputs, 2.)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            owner = torch.add(inputs, 2.)
            output = extension.weak_alias(owner)
            owner_ref = weakref.ref(owner)
            del owner
        assert owner_ref() is None
        inputs.add_(1.)
        for _ in range(10):
            graph.replay()
            assert torch.equal(output, inputs + 2.)
            inputs.add_(.125)
        live.append((graph, inputs, output))
        result["graph_cases"].append(dict(shape=shape, owner_released=True, changed_input_replays=10, bitwise_exact=True))
        save()
    # Shared-pool graphs are replayed in the same order as capture; each output is consumed immediately.
    for graph, inputs, output in live:
        graph.replay()
        assert torch.equal(output, inputs + 2.)
    result.update(complete=True, CPU_owner_lifetime_verified=True, changed_input_graphs_bitwise_verified=True,
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), Path(__file__).with_name("weak_alias.py"), Path(__file__).with_name("weak_alias.cpp"))})
    save()
    print(json.dumps(result))


if __name__ == "__main__": main()
