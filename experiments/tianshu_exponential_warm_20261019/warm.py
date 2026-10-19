# SPDX-License-Identifier: Apache-2.0
"""Warm the actual contiguous FP32 sampler noise shapes without advancing RNG state."""
import time


def warm_exponential(torch, device='cuda:0', max_batch=64, vocab=130560):
    assert max_batch==64 and vocab==130560
    assert not torch.cuda.is_current_stream_capturing(), 'Warm RNG eagerly, outside graphs'
    index=torch.device(device).index
    if index is None:
        index=torch.cuda.current_device()
    started=time.perf_counter()
    shapes=[]
    # fork_rng preserves CPU and the selected default device generator; this
    # helper never receives or touches any per-request Generator objects.
    with torch.random.fork_rng(devices=[index], enabled=True):
        noise=torch.empty((max_batch,vocab),dtype=torch.float32,device=device)
        for batch in range(max_batch,0,-1):
            view=noise[:batch]
            assert view.is_contiguous() and view.numel()==batch*vocab
            view.exponential_()
            shapes.append(dict(batch=batch,elements=batch*vocab))
        torch.cuda.synchronize(device)
        del view,noise
    return dict(complete=True,shapes=shapes,maximum_noise_bytes=max_batch*vocab*4,
        elapsed_seconds=time.perf_counter()-started,
        dispatch_context_must_be_established_by_caller=True,
        no_rng_operation_captured_in_graph=True)
