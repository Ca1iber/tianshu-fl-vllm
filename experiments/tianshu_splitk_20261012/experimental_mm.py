# SPDX-License-Identifier: Apache-2.0
"""NT BF16 decode GEMM prototype with FP32 partials, no atomic BF16 sum."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(a, weight, partial, out, m: tl.constexpr, n: tl.constexpr,
          k: tl.constexpr, bm: tl.constexpr, bn: tl.constexpr,
          bk: tl.constexpr, splits: tl.constexpr):
    mi = tl.program_id(0) * bm + tl.arange(0, bm)
    ni = tl.program_id(1) * bn + tl.arange(0, bn)
    segment = tl.program_id(2)
    ki = segment * bk + tl.arange(0, bk)
    ap = a + mi[:, None] * k + ki[None, :]
    bp = weight + ni[None, :] * k + ki[:, None]
    acc = tl.zeros((bm, bn), tl.float32)
    for step in range(tl.cdiv(k, bk * splits)):
        av = tl.load(ap, mi[:, None] < m, other=0)
        bv = tl.load(bp, ni[None, :] < n, other=0)
        acc = tl.dot(av, bv, acc, input_precision=None)
        ap += bk * splits
        bp += bk * splits
    offset = mi[:, None] * n + ni[None, :]
    mask = (mi[:, None] < m) & (ni[None, :] < n)
    if splits == 1:
        tl.store(out + offset, acc, mask)
    else:
        tl.store(partial + segment * m * n + offset, acc, mask)


@triton.jit
def _sum(partial, out, count: tl.constexpr, splits: tl.constexpr, block: tl.constexpr):
    off = tl.program_id(0) * block + tl.arange(0, block)
    acc = tl.zeros((block,), tl.float32)
    for segment in range(splits):
        acc += tl.load(partial + segment * count + off, off < count, other=0)
    tl.store(out + off, acc, off < count)


def prepare(a, weight, config):
    assert a.ndim == weight.ndim == 2 and a.shape[1] == weight.shape[1]
    assert a.dtype == weight.dtype == torch.bfloat16 and a.is_contiguous() and weight.is_contiguous()
    assert a.shape[1] % (config['bk'] * config['splits']) == 0
    m, k = a.shape
    n = weight.shape[0]
    return (torch.empty((config['splits'], m, n), device=a.device, dtype=torch.float32),
            torch.empty((m, n), device=a.device, dtype=a.dtype))


def mm(a, weight, scratch, config):
    m, k = a.shape
    n = weight.shape[0]
    partial, out = scratch
    _gemm[(triton.cdiv(m, config['bm']), triton.cdiv(n, config['bn']), config['splits'])](
        a, weight, partial, out, m, n, k, config['bm'], config['bn'], config['bk'],
        config['splits'], num_warps=config['warps'], num_stages=config['stages'])
    if config['splits'] != 1:
        _sum[(triton.cdiv(m * n, 1024),)](partial, out, m * n, config['splits'], 1024)
    return out
