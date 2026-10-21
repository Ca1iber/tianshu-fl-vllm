# SPDX-License-Identifier: Apache-2.0
"""Matched-layout GEMM control, adapted from MetaX's packed NT experiment.

PACKED changes only weight addressing. Both paths use identical tiles,
launch order, FP32 accumulation and BF16 stores. This is not dispatched by
FlagGems or vLLM. Callers own immutable weights and output allocations.
"""

import triton
import triton.language as tl


@triton.jit
def matched_nt_kernel(
    A, W, C, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, PACKED: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
):
    pid = tl.program_id(0)
    gm, gn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid // (GROUP_M * gn)
    size = tl.minimum(gm - group * GROUP_M, GROUP_M)
    pm = group * GROUP_M + pid % size
    pn = pid % (GROUP_M * gn) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    am = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
    wn = tl.max_contiguous(tl.multiple_of(rn, BN), BN)
    rk = tl.arange(0, BK)
    ap = A + am[:, None] * K + rk[None, :]
    if PACKED:
        wp = W + pn * (K // BK) * BN * BK + tl.arange(0, BN)[None, :] * BK + rk[:, None]
    else:
        wp = W + wn[None, :] * K + rk[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(tl.cdiv(K, BK)):
        if EVEN_M:
            a = tl.load(ap)
        else:
            a = tl.load(ap, mask=rm[:, None] < M, other=0)
        if EVEN_N:
            b = tl.load(wp)
        else:
            b = tl.load(wp, mask=rn[None, :] < N, other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        ap += BK
        if PACKED:
            wp += BN * BK
        else:
            wp += BK
    if EVEN_M and EVEN_N:
        tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))
    else:
        tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


def launch(x, weight, out, config, packed):
    m, k = x.shape
    n = out.shape[1]
    bm, bn, bk, warps, stages = config
    matched_nt_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        x, weight, out, m, n, k,
        BM=bm, BN=bn, BK=bk, GROUP_M=8, PACKED=packed,
        EVEN_M=m % bm == 0, EVEN_N=n % bn == 0,
        num_warps=warps, num_stages=stages,
    )
    return out
