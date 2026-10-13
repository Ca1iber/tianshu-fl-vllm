# SPDX-License-Identifier: Apache-2.0
"""Paired Gate/Up NT BF16 GEMM with the compiled model's SiLU epilogue."""
import torch
import triton
import triton.language as tl


@triton.jit
def _paired_gateup(A, W, O, M: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
                   OS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr):
    # BN is the physical projection width; each CTA computes BN/2 paired
    # gate and up columns without changing the stored weight layout.
    grid_n = tl.cdiv(I, BN // 2)
    pid = tl.program_id(0)
    pid_m, pid_n = pid // grid_n, pid % grid_n
    rows = pid_m * BM + tl.arange(0, BM)
    columns = tl.arange(0, BN)
    local = columns % (BN // 2)
    logical = pid_n * (BN // 2) + local
    physical = logical + tl.where(columns < BN // 2, 0, I)
    kk = tl.arange(0, BK)
    ap = A + rows[:, None] * K + kk[None, :]
    wp = W + physical[None, :] * K + kk[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(K // BK):
        if M % BM == 0:
            a = tl.load(ap)
        else:
            a = tl.load(ap, rows[:, None] < M, other=0)
        if I % (BN // 2) == 0:
            w = tl.load(wp)
        else:
            w = tl.load(wp, logical[None, :] < I, other=0)
        acc = tl.dot(a, w, acc, input_precision=None)
        ap += BK
        wp += BK
    # Original GateUp materializes BF16 before the compiled SiLU/multiply
    # kernel. That kernel does SiLU and multiplication in FP32, with no
    # intervening BF16 SiLU store. Preserve both rounding boundaries.
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    paired = tl.trans(tl.reshape(rounded, (BM, 2, BN // 2)), (0, 2, 1))
    gate, up = tl.split(paired)
    result = (gate * tl.sigmoid(gate)) * up
    out_columns = pid_n * (BN // 2) + tl.arange(0, BN // 2)
    tl.store(O + rows[:, None] * OS + out_columns[None, :], result,
             (rows[:, None] < M) & (out_columns[None, :] < I))


@triton.jit
def _affine_gateup(A,W,O,M:tl.constexpr,I:tl.constexpr,K:tl.constexpr,OS:tl.constexpr,
                  BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    # Keep two affine weight loads and two accumulators. This removes the
    # non-affine concatenated column map and accumulator reshape/split path
    # from the failed first prototype, without changing projection rounding.
    half: tl.constexpr = BN//2
    grid_n=tl.cdiv(I,half);pid=tl.program_id(0)
    rows=(pid//grid_n)*BM+tl.arange(0,BM)
    columns=(pid%grid_n)*half+tl.arange(0,half)
    kk=tl.arange(0,BK)
    ap=A+rows[:,None]*K+kk[None,:]
    gatep=W+columns[None,:]*K+kk[:,None]
    upp=W+(columns[None,:]+I)*K+kk[:,None]
    gate_acc=tl.zeros((BM,half),tl.float32);up_acc=tl.zeros((BM,half),tl.float32)
    for step in range(K//BK):
        a=tl.load(ap,mask=rows[:,None]<M,other=0)
        gatew=tl.load(gatep,mask=columns[None,:]<I,other=0)
        upw=tl.load(upp,mask=columns[None,:]<I,other=0)
        gate_acc=tl.dot(a,gatew,gate_acc,input_precision=None)
        up_acc=tl.dot(a,upw,up_acc,input_precision=None)
        ap+=BK;gatep+=BK;upp+=BK
    gate=gate_acc.to(tl.bfloat16).to(tl.float32)
    up=up_acc.to(tl.bfloat16).to(tl.float32)
    value=(gate*tl.sigmoid(gate))*up
    tl.store(O+rows[:,None]*OS+columns[None,:],value,
             mask=(rows[:,None]<M)&(columns[None,:]<I))


def launch(x, weight, output, config):
    assert x.ndim == weight.ndim == output.ndim == 2
    assert x.dtype == weight.dtype == output.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous()
    m, k = x.shape
    assert weight.shape[1] == k and weight.shape[0] % 2 == 0
    inter = weight.shape[0] // 2
    assert output.shape[0] >= m and output.shape[1] >= inter and output.stride(1) == 1
    assert k % config['bk'] == 0 and x.device == weight.device == output.device
    grid = (triton.cdiv(m, config['bm']) * triton.cdiv(inter, config['bn']//2),)
    kernel=_affine_gateup if config.get('affine',False) else _paired_gateup
    kernel[grid](x, weight, output, m, inter, k, output.stride(0),
                        config['bm'], config['bn'], config['bk'],
                        num_warps=config['warps'], num_stages=config['stages'])
    return output
