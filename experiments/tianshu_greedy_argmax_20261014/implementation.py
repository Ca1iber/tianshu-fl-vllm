# SPDX-License-Identifier: Apache-2.0
"""Two-stage exact greedy argmax; first-index ties and first NaN precedence."""
import torch
import triton
import triton.language as tl


@triton.jit
def partial_max(X, V, I, N: tl.constexpr, XS: tl.constexpr, TILES: tl.constexpr, TILE: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    col = block * TILE + tl.arange(0, TILE)
    valid = col < N
    value = tl.load(X + row * XS + col, valid, other=-float("inf"))
    nan = valid & (value != value)
    first_nan = tl.min(tl.where(nan, col, 2147483647), 0)
    maximum = tl.max(tl.where(nan, -float("inf"), value), 0)
    index = tl.min(tl.where(valid & (value == maximum), col, 2147483647), 0)
    has_nan = first_nan != 2147483647
    tl.store(V + row * TILES + block, tl.where(has_nan, float("nan"), maximum))
    tl.store(I + row * TILES + block, tl.where(has_nan, first_nan, index))


@triton.jit
def finish_max(V, I, O, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < TILES
    value = tl.load(V + row * TILES + col, valid, other=-float("inf"))
    index = tl.load(I + row * TILES + col, valid, other=2147483647)
    nan = valid & (value != value)
    first_nan = tl.min(tl.where(nan, index, 2147483647), 0)
    maximum = tl.max(tl.where(nan, -float("inf"), value), 0)
    first_max = tl.min(tl.where(valid & (value == maximum), index, 2147483647), 0)
    result = tl.where(first_nan != 2147483647, first_nan, first_max)
    tl.store(O + row, result.to(tl.int64))


def launch(logits, config):
    assert logits.ndim == 2 and logits.dtype == torch.float32 and logits.is_cuda
    assert logits.stride(1) == 1 and logits.shape[1] > 0
    rows, vocab = logits.shape
    tile = config["tile"]
    tiles = triton.cdiv(vocab, tile)
    values = torch.empty((rows, tiles), device=logits.device, dtype=torch.float32)
    indices = torch.empty((rows, tiles), device=logits.device, dtype=torch.int32)
    output = torch.empty((rows,), device=logits.device, dtype=torch.int64)
    partial_max[(rows, tiles)](logits, values, indices, vocab, logits.stride(0), tiles, tile,
                              num_warps=config["warps"], num_stages=1)
    finish_max[(rows,)](values, indices, output, tiles, triton.next_power_of_2(tiles), num_warps=4, num_stages=1)
    return output
