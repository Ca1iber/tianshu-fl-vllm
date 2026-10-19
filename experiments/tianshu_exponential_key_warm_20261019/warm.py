# SPDX-License-Identifier: Apache-2.0
"""Warm four actual scalar-type entries; restore the full default RNG state."""
import time

KEYS = tuple((seed, offset) for seed in (7, 2**32 + 7) for offset in (0, 2**32))


def warm_exponential(torch, set_state, device='cuda:0', max_batch=64, vocab=130560):
    assert max_batch == 64 and vocab == 130560
    assert not torch.cuda.is_current_stream_capturing()
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    started = time.perf_counter()
    calls = []
    with torch.random.fork_rng(devices=[index], enabled=True):
        noise = torch.empty((max_batch, vocab), dtype=torch.float32, device=device)
        for seed, offset in KEYS:
            set_state(seed, offset, index)
            noise.exponential_()
            calls.append(dict(seed=seed, offset=offset, batch=max_batch, elements=max_batch*vocab))
        torch.cuda.synchronize(device)
        del noise
    return dict(complete=True, calls=calls, maximum_noise_bytes=max_batch*vocab*4,
        elapsed_seconds=time.perf_counter()-started,
        no_rng_operation_captured_in_graph=True)
