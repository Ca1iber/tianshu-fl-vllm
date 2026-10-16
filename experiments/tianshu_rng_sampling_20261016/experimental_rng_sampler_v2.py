# SPDX-License-Identifier: Apache-2.0
"""Experimental FP32 exponential-race sampler with in-kernel Philox draws.

The unseeded eager path consumes fresh, disjoint 64-bit Philox counters.
Its global random sequence differs from Torch/FlagGems exponential_. Per-request
generators and capture must use the original sampler. No TopP change is made.
"""
import torch
import triton
import triton.language as tl
from flag_gems.utils.random_utils import philox_backend_seed_offset, uint_to_uniform_float


@triton.jit
def _words(seed, offset, count: tl.constexpr, output, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    counter = offset.to(tl.uint64) + i.to(tl.uint64)
    c0 = counter.to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    z = tl.full((block,), 0, tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed.to(tl.uint64), c0, c1, z, z)
    tl.store(output + i, r0.to(tl.int64), i < count)
    tl.store(output + count + i, r1.to(tl.int64), i < count)
    tl.store(output + 2 * count + i, r2.to(tl.int64), i < count)
    tl.store(output + 3 * count + i, r3.to(tl.int64), i < count)


@triton.jit
def _exponential(word):
    u = uint_to_uniform_float(word)
    # Match the FP32 exponential transform's positive upper-end clamp.
    return -tl.where(u >= 1.0 - 5.960464477539063e-8,
                     -5.960464477539063e-8, tl.log(u))


@triton.jit
def _partial(logits, noise, scores, indices, stride: tl.constexpr,
             vocab: tl.constexpr, tiles: tl.constexpr, seed, offset,
             quarter: tl.constexpr, materialize: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    lane = tl.arange(0, quarter)
    counter = (offset.to(tl.uint64) + row.to(tl.uint64) * tiles * quarter
               + tile.to(tl.uint64) * quarter + lane.to(tl.uint64))
    c0 = counter.to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    z = tl.full((quarter,), 0, tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed.to(tl.uint64), c0, c1, z, z)
    best = tl.full((), -float('inf'), tl.float32)
    index = tl.full((), 2147483647, tl.int32)
    invalid = tl.full((), False, tl.int1)
    for part in tl.static_range(4):
        if part == 0:
            q = _exponential(r0)
        elif part == 1:
            q = _exponential(r1)
        elif part == 2:
            q = _exponential(r2)
        else:
            q = _exponential(r3)
        off = tile * (4 * quarter) + part * quarter + lane
        x = tl.load(logits + row * stride + off, off < vocab,
                    other=-float('inf'))
        if materialize:
            tl.store(noise + row * vocab + off, q, off < vocab)
        value = x - tl.log(q)
        value = tl.where((off < vocab) & (x == x) & (x != -float('inf')),
                         value, -float('inf'))
        this_best = tl.max(value, 0)
        this_index = tl.min(tl.where((off < vocab) & (value == this_best),
                                     off, 2147483647), 0)
        index = tl.where(this_best > best, this_index,
                         tl.where(this_best == best, tl.minimum(index, this_index), index))
        best = tl.maximum(best, this_best)
        invalid = invalid | (tl.sum((x == float('inf')).to(tl.int32), 0) > 0)
    # At model vocab width, FlagGems ignores isolated NaNs in its split
    # softmax/argmax path. A +inf row instead yields all-NaN probabilities
    # and index0. The v1 diagnostic records both observed cases.
    tl.store(scores + row * tiles + tile, tl.where(invalid, float('inf'), best))
    tl.store(indices + row * tiles + tile, tl.where(invalid, -1, index))


@triton.jit
def _finish(scores, indices, output, tiles: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    off = tl.arange(0, block)
    value = tl.load(scores + row * tiles + off, off < tiles, other=-float('inf'))
    index = tl.load(indices + row * tiles + off, off < tiles, other=2147483647)
    invalid = tl.sum((index < 0).to(tl.int32), 0) > 0
    best = tl.max(value, 0)
    result = tl.min(tl.where(value == best, index, 2147483647), 0)
    tl.store(output + row, tl.where(invalid, 0, result).to(tl.int64))


def prepare(logits, block=2048):
    assert logits.ndim == 2 and logits.dtype == torch.float32
    assert logits.device.type == 'cuda' and logits.stride(1) == 1
    assert block in (1024, 2048, 4096)
    rows, vocab = logits.shape
    assert rows > 0 and vocab > 0
    tiles = triton.cdiv(vocab, block)
    return (torch.empty((rows, tiles), device=logits.device, dtype=torch.float32),
            torch.empty((rows, tiles), device=logits.device, dtype=torch.int32),
            torch.empty((rows,), device=logits.device, dtype=torch.int64))


def reservation_size(logits, block=2048):
    return logits.shape[0] * triton.cdiv(logits.shape[1], block) * (block // 4)


def sample_seeded(logits, scratch, seed, offset, block=2048, noise=None):
    """Kernel-only deterministic primitive; callers own counter reservation.

    A fixed seed/offset captured in a graph repeats the same draw. This is used
    only in the standalone numerical/timing probe, not the service adapter.
    """
    scores, indices, output = scratch
    rows, vocab = logits.shape
    tiles = triton.cdiv(vocab, block)
    assert scores.shape == indices.shape == (rows, tiles)
    assert output.shape == (rows,)
    assert 0 <= offset <= (1 << 63) - 1 - reservation_size(logits, block)
    if noise is not None:
        assert noise.shape == logits.shape and noise.is_contiguous()
        assert noise.dtype == torch.float32 and noise.device == logits.device
    _partial[(rows, tiles)](logits, logits if noise is None else noise,
        scores, indices, logits.stride(0), vocab, tiles, seed, offset,
        block // 4, noise is not None, num_warps=4, num_stages=1)
    _finish[(rows,)](scores, indices, output, tiles,
                    triton.next_power_of_2(tiles), num_warps=4)
    return output


def eligible(logits, generators, use_fp64_gumbel=False):
    return (not generators and not use_fp64_gumbel and logits.ndim == 2
            and 0 < logits.shape[0] <= 64 and logits.shape[1] == 130560
            and logits.dtype == torch.float32 and logits.stride(1) == 1
            and logits.device.type == 'cuda'
            and not torch.cuda.is_current_stream_capturing())


def sample_global(logits, block=2048):
    """Eager unseeded operation, including real RNG-state and scratch costs."""
    assert eligible(logits, {})
    scratch = prepare(logits, block)
    seed, offset = philox_backend_seed_offset(reservation_size(logits, block))
    return sample_seeded(logits, scratch, seed, offset, block)


def random_words(seed, offset, count, device='cuda'):
    output = torch.empty((4, count), device=device, dtype=torch.int64)
    _words[(triton.cdiv(count, 128),)](seed, offset, count, output, 128)
    return output
