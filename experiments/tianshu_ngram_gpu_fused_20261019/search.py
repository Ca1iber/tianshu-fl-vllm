# SPDX-License-Identifier: Apache-2.0
"""Two-pass integer suffix matching; not yet GPU validated or installed."""
import torch
import triton
import triton.language as tl


@triton.jit
def _partials(IDS, LENGTHS, ENABLED, SCORES,
              ROW: tl.constexpr, LS: tl.constexpr, MS: tl.constexpr,
              CAP: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    length = tl.load(LENGTHS + row * LS).to(tl.int32)
    enabled = tl.load(ENABLED + row * MS)
    invalid = 3 * (CAP + 1)
    best = tl.full((), invalid, tl.int32)
    if enabled and length >= 4:
        base = row * ROW
        s0 = tl.load(IDS + base + length - 1)
        s1 = tl.load(IDS + base + length - 2)
        s2 = tl.load(IDS + base + length - 3)
        s3 = tl.load(IDS + base + length - 4, mask=length >= 4, other=0)
        s4 = tl.load(IDS + base + length - 5, mask=length >= 5, other=0)
        tiles = tl.cdiv(length, PARTS * BLOCK)
        start = part * tiles * BLOCK
        end = tl.minimum(start + tiles * BLOCK, length - 1)
        offsets = tl.arange(0, BLOCK)
        for block in range(start, end, BLOCK):
            position = block + offsets
            valid = (position >= 2) & (position < end)
            a0 = tl.load(IDS + base + position, mask=valid, other=0)
            a1 = tl.load(IDS + base + position - 1, mask=valid, other=0)
            a2 = tl.load(IDS + base + position - 2, mask=valid, other=0)
            match3 = valid & (a0 == s0) & (a1 == s1) & (a2 == s2)
            a3 = tl.load(IDS + base + position - 3,
                         mask=match3 & (position >= 3), other=0)
            match4 = match3 & (position >= 3) & (length >= 4) & (a3 == s3)
            a4 = tl.load(IDS + base + position - 4,
                         mask=match4 & (position >= 4) & (length >= 5), other=0)
            match5 = match4 & (position >= 4) & (length >= 5) & (a4 == s4)
            matched = tl.where(match5, 5, tl.where(match4, 4, 3))
            first = position - matched + 1
            score = (5 - matched) * (CAP + 1) + first
            best = tl.minimum(best, tl.min(tl.where(match3, score, invalid), axis=0))
    tl.store(SCORES + row * PARTS + part, best)


@triton.jit
def _finish(IDS, SCORES, DRAFT, COUNTS,
            ROW: tl.constexpr, CAP: tl.constexpr, PARTS: tl.constexpr):
    row = tl.program_id(0)
    scores = tl.load(SCORES + row * PARTS + tl.arange(0, PARTS))
    best = tl.min(scores, axis=0)
    valid = best < 3 * (CAP + 1)
    matched = 5 - best // (CAP + 1)
    first = best % (CAP + 1)
    token = tl.load(IDS + row * ROW + first + matched, mask=valid, other=-1)
    tl.store(DRAFT + row, token)
    tl.store(COUNTS + row, (valid & (token != -1)).to(tl.int64))


def forward(lengths, token_ids, enabled, *, parts=8, block=256):
    assert parts in (4, 8, 16) and block in (128, 256, 512)
    assert token_ids.is_cuda and token_ids.ndim == 2 and token_ids.stride(1) == 1
    assert token_ids.dtype in (torch.int32, torch.int64)
    n, cap = token_ids.shape
    assert 1 <= n <= 256 and 5 <= cap <= 131072
    assert (n - 1) * token_ids.stride(0) + cap < 2**31
    assert lengths.shape == enabled.shape == (n,)
    assert lengths.is_cuda and enabled.is_cuda and enabled.dtype == torch.bool
    assert lengths.dtype in (torch.int32, torch.int64)
    scores = torch.empty((n, parts), device=token_ids.device, dtype=torch.int32)
    draft = torch.empty((n, 1), device=token_ids.device, dtype=token_ids.dtype)
    counts = torch.empty((n,), device=token_ids.device, dtype=torch.int64)
    _partials[(n, parts)](token_ids, lengths, enabled, scores,
        ROW=token_ids.stride(0), LS=lengths.stride(0), MS=enabled.stride(0),
        CAP=cap, PARTS=parts, BLOCK=block, num_warps=4, num_stages=1)
    _finish[(n,)](token_ids, scores, draft, counts,
                  ROW=token_ids.stride(0), CAP=cap, PARTS=parts, num_warps=4)
    return draft, counts
