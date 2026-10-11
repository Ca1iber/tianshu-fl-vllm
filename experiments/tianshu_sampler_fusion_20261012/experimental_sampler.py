# SPDX-License-Identifier: Apache-2.0
"""Experimental exponential-race sampling with unchanged noise generation.

For positive noise q, argmax(softmax(logits)/q) equals argmax(logits-log(q))
in real arithmetic. FP32 boundary behavior is tested independently; this is
not an assertion of bitwise equivalence for every possible floating input.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(logits, noise, scores, indices, stride_l: tl.constexpr,
             stride_q: tl.constexpr, vocab: tl.constexpr, tiles: tl.constexpr,
             block: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    off = tile * block + tl.arange(0, block)
    x = tl.load(logits + row * stride_l + off, off < vocab, other=-float('inf'))
    q = tl.load(noise + row * stride_q + off, off < vocab, other=1.0)
    value = x - tl.log(q)
    # Masked logits never win, including the rare zero-noise case.
    value = tl.where((off < vocab) & (x != -float('inf')), value, -float('inf'))
    best = tl.max(value, 0)
    index = tl.min(tl.where(value == best, off, 2147483647), 0)
    tl.store(scores + row * tiles + tile, best)
    tl.store(indices + row * tiles + tile, index)


@triton.jit
def _finish(scores, indices, output, tiles: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    off = tl.arange(0, block)
    value = tl.load(scores + row * tiles + off, off < tiles, other=-float('inf'))
    index = tl.load(indices + row * tiles + off, off < tiles, other=2147483647)
    best = tl.max(value, 0)
    result = tl.min(tl.where(value == best, index, 2147483647), 0)
    tl.store(output + row, result.to(tl.int64))


def prepare(logits, block=2048):
    assert logits.ndim == 2 and logits.dtype == torch.float32 and logits.stride(1) == 1
    rows, vocab = logits.shape
    tiles = triton.cdiv(vocab, block)
    return (torch.empty((rows, tiles), device=logits.device, dtype=torch.float32),
            torch.empty((rows, tiles), device=logits.device, dtype=torch.int32),
            torch.empty((rows,), device=logits.device, dtype=torch.int64))


def sample_with_noise(logits, noise, scratch, block=2048):
    assert noise.shape == logits.shape and noise.dtype == torch.float32
    assert noise.stride(1) == 1 and noise.device == logits.device
    scores, indices, output = scratch
    rows, vocab = logits.shape
    tiles = triton.cdiv(vocab, block)
    assert scores.shape == indices.shape == (rows, tiles) and output.shape == (rows,)
    if rows:
        _partial[(rows, tiles)](logits, noise, scores, indices, logits.stride(0),
                               noise.stride(0), vocab, tiles, block, num_warps=4)
        _finish[(rows,)](scores, indices, output, tiles, triton.next_power_of_2(tiles), num_warps=4)
    return output
