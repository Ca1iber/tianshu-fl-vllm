# SPDX-License-Identifier: Apache-2.0
"""Original A8 arithmetic; vary independent rows per CTA and warp layout."""
import torch
import triton
import triton.language as tl
from triton.language.extra.corex import libdevice

@triton.jit
def _quantize(A, Q, S, M: tl.constexpr, K: tl.constexpr, ASTRIDE: tl.constexpr,
              BLOCK: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    offsets = tl.arange(0, BLOCK)
    value = tl.load(A + rows[:, None] * ASTRIDE + offsets[None, :],
                    (rows[:, None] < M) & (offsets[None, :] < K), other=0).to(tl.float32)
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(value), 1), 1.e-8), 127.)
    packed = libdevice.nearbyint(tl.div_rn(value, scale[:, None]))
    packed = tl.minimum(tl.maximum(packed, -127.), 127.).to(tl.int8)
    tl.store(Q + rows[:, None] * K + offsets[None, :], packed,
             (rows[:, None] < M) & (offsets[None, :] < K))
    tl.store(S + rows, scale, rows < M)

def quantize(a, q, scales, rows, warps):
    m, k = a.shape
    assert a.dtype == torch.bfloat16 and q.dtype == torch.int8 and scales.dtype == torch.float32
    assert a.stride(1) == 1 and a.stride(0) >= k
    assert q.shape == a.shape and q.is_contiguous() and scales.shape == (m,) and scales.is_contiguous()
    assert rows in (1, 2, 4, 8) and warps in (1, 2, 4, 8)
    return _quantize[(triton.cdiv(m, rows),)](a, q, scales, m, k, a.stride(0),
        triton.next_power_of_2(k), rows, num_warps=warps, num_stages=1)
