# SPDX-License-Identifier: Apache-2.0
"""Repeat the exact old O8 fixture, retaining full original BF16 output variability."""
import hashlib
from pathlib import Path

def run(a,w,cpu_a,cpu_w,baseline,root,torch):
    assert tuple(a.shape)==(8,2048) and tuple(w.shape)==(2048,2048)
    a.copy_(cpu_a);w.copy_(cpu_w)
    graph=torch.cuda.CUDAGraph()
    for _ in range(2):output=baseline.qualified_mm(a,w)
    with torch.cuda.graph(graph):output=baseline.qualified_mm(a,w)
    rng=torch.cuda.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone();values=[]
    reference=cpu_a.float()@cpu_w.float().T
    for _ in range(32):
        graph.replay();torch.cuda.synchronize();values.append(output.cpu().clone())
    assert torch.equal(torch.cuda.get_rng_state(),rng) and torch.equal(torch.get_rng_state(),cpu_rng)
    assert torch.equal(a.cpu(),cpu_a) and torch.equal(w.cpu(),cpu_w)
    stacked=torch.stack(values);error=(stacked.float()-reference[None]).abs()
    limit=.02+.01*reference.abs()
    bad=error>limit
    path=Path(root)/'native_O_M8_repeat.pt'
    torch.save(dict(a=cpu_a,outputs=stacked,CPU_FP32_reference=reference,RNG_unchanged=True),path)
    return dict(replays=32,full_output_values=stacked.numel(),unique_full_output_hashes=len({hashlib.sha256(memoryview(v.contiguous().view(torch.uint8).numpy())).hexdigest() for v in values}),
        all_original_tolerance_passed=not bool(bad.any()),replays_with_exceedance=int(bad.flatten(1).any(1).sum()),
        total_elements_exceeding_original_tolerance=int(bad.sum()),max_absolute_error=float(error.max()),
        max_excess_over_original_tolerance=float((error-limit[None]).max()),raw_path=str(path),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original_tolerance_unchanged=True)
