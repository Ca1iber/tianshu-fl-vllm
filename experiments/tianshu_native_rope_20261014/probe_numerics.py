# SPDX-License-Identifier: Apache-2.0
"""Exact native BF16 RoPE checks on packed QKV strides and cache boundaries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); root = args.output.resolve(); assert not root.exists()
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    deadline = datetime.fromisoformat(json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())["stop_at_utc"])
    assert (deadline - datetime.now(timezone.utc)).total_seconds() > 600
    signal.alarm(600); root.mkdir(parents=True)
    import torch
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    import sys
    impl = sys.modules["_tianshu_native_rope_impl"]
    original = RotaryEmbedding._tianshu_original_forward_static
    assert os.environ["TIANSHU_NATIVE_ROPE_RECORD_DIR"] == str(root / "records")
    for path in [Path(__file__), Path(__file__).with_name("implementation.py"), Path(__file__).with_name("hooks") / "sitecustomize.py"]:
        (root / path.name).write_bytes(path.read_bytes())
    data = dict(complete=False, numerical_checks_passed=False, official_benchmark=False, adopted=False,
                source_sha256=hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest(), cases=[])
    torch.manual_seed(2026101494)
    cache = torch.randn(131072, 128, device="cuda", dtype=torch.bfloat16)
    cpu_cache = cache.cpu()
    with torch.inference_mode():
        for rows in (1, 7, 64, 576, 2048):
            for packed in (False, True):
                for distribution in ("random", "zero", "large", "small"):
                    if packed:
                        qkv = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
                        q = qkv[:, :2048]; k = qkv[:, 2048:2304]
                    else:
                        q = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
                        k = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
                    if distribution == "zero":q.zero_(); k.zero_()
                    elif distribution == "large":q.mul_(100); k.mul_(100)
                    elif distribution == "small":q.mul_(.001); k.mul_(.001)
                    positions = torch.arange(rows, device="cuda", dtype=torch.int64).remainder(131072)
                    positions[0] = 131071
                    if rows > 1:positions[1] = 0
                    if rows > 2:positions[2] = 4095
                    expected_q, expected_k = original(positions, q, k, 128, 128, cache, True)
                    saved_q = q.clone(); saved_k = k.clone()
                    actual_q, actual_k = RotaryEmbedding.forward_static(positions, q, k, 128, 128, cache, True)
                    assert torch.equal(actual_q, expected_q) and torch.equal(actual_k, expected_k), (rows, packed, distribution)
                    assert torch.equal(q, saved_q) and torch.equal(k, saved_k)
                    assert actual_q.data_ptr() not in (q.data_ptr(), k.data_ptr()) and actual_k.data_ptr() not in (q.data_ptr(), k.data_ptr(), actual_q.data_ptr())
                    # Independent CPU reference with explicit BF16 product rounding.
                    selected = cpu_cache[positions.cpu()]; cosine, sine = selected.chunk(2, dim=-1)
                    def cpu_reference(x, heads):
                        xx = x.cpu().reshape(rows, heads, 128); a, b = xx.chunk(2, dim=-1)
                        co = cosine[:, None, :]; si = sine[:, None, :]
                        y1 = ((a.float()*co.float()).bfloat16().float()-(b.float()*si.float()).bfloat16().float()).bfloat16()
                        y2 = ((b.float()*co.float()).bfloat16().float()+(a.float()*si.float()).bfloat16().float()).bfloat16()
                        return torch.cat((y1,y2),dim=-1).reshape(x.shape)
                    assert torch.equal(actual_q.cpu(), cpu_reference(q,16)) and torch.equal(actual_k.cpu(), cpu_reference(k,2))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):gq, gk = RotaryEmbedding.forward_static(positions, q, k, 128, 128, cache, True)
                    graph.replay(); torch.cuda.synchronize()
                    assert torch.equal(gq, actual_q) and torch.equal(gk, actual_k)
                    data["cases"].append(dict(rows=rows, packed_qkv=packed, distribution=distribution,
                                              exact_native=True, exact_cpu=True, inputs_unchanged=True,
                                              alias_correct=True, graph_correct=True))
                    (root / "result.json").write_text(json.dumps(data, indent=2)+"\n")
        # Verify the method keeps original handling when key is absent.
        p = torch.tensor([0], device="cuda", dtype=torch.int64); q = torch.randn(1,2048,device="cuda",dtype=torch.bfloat16)
        aq, ak = RotaryEmbedding.forward_static(p,q,None,128,128,cache,True); eq, ek = original(p,q,None,128,128,cache,True)
        assert ak is ek is None and torch.equal(aq,eq)
        assert torch.equal(cache.cpu(), cpu_cache)
    records=[json.loads(p.read_text()) for p in (root/"records").glob("*.json")]
    assert records and any(x["query_row_stride"]==2560 for x in records)
    data.update(complete=True,numerical_checks_passed=True,key_none_fallback_verified=True,actual_records=records)
    (root/"result.json").write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({k:v for k,v in data.items() if k not in ("cases","actual_records")}))


if __name__=="__main__":main()
