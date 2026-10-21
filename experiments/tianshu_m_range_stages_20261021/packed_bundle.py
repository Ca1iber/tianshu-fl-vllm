# SPDX-License-Identifier: Apache-2.0
"""Variable-row BF16 packed kernels with fixed MiniCPM5-2B channels.

Weights and workspaces are caller-owned. These inference APIs have no model,
weight, or workspace caches, and require warmup before graph capture.
"""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

from .packed_swiglu import (
    INPUT_PACKED_SHAPE,
    WEIGHT_PACKED_SHAPE,
    _check_tensor,
    _input_rows,
    _overlaps,
    pack_gate_up_input,
)

# Legacy M2048 allocation constants; APIs derive actual rows from x.
M, D, H = 2048, 2048, 6144
ACTIVATION_PACKED_SHAPE = (96, 2048, 64)
DOWN_PACKED_SHAPE = (32, 96, 64, 64)
LM_HEAD_PACKED_SHAPE = (1020, 64, 128, 32)
LM_HEAD_N = 130560


@triton.jit
def _packed_nt_kernel(
    A, B, C, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, PACK_A: tl.constexpr,
    EVEN_M: tl.constexpr = False,
):
    pid = tl.program_id(0)
    gm, gn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    width = GROUP_M * gn
    group = pid // width
    size = tl.minimum(gm - group * GROUP_M, GROUP_M)
    pm = group * GROUP_M + pid % size
    pn = pid % width // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    am = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
    rk = tl.arange(0, BK)
    if PACK_A:
        ap = A + am[:, None] * BK + rk[None, :]
    else:
        ap = A + am[:, None] * K + rk[None, :]
    bp = (
        B + pn * (K // BK) * BN * BK
        + tl.arange(0, BN)[None, :] * BK + rk[:, None]
    )
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(tl.cdiv(K, BK)):
        if EVEN_M:
            a = tl.load(ap)
        else:
            a = tl.load(ap, mask=rm[:, None] < M, other=0.0)
        b = tl.load(bp)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        if PACK_A:
            ap += M * BK
        else:
            ap += BK
        bp += BN * BK
    if EVEN_M:
        tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))
    else:
        tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
                 mask=rm[:, None] < M)


@triton.jit
def _packed_gate_up_silu_kernel(
    XP, WB, Z, M, H, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr = False,
):
    pid = tl.program_id(0)
    gm, gn = tl.cdiv(M, BM), tl.cdiv(H, BN)
    group = pid // (GROUP_M * gn)
    size = tl.minimum(gm - group * GROUP_M, GROUP_M)
    pm = group * GROUP_M + pid % size
    pn = pid % (GROUP_M * gn) // size
    rm = pm * BM + tl.arange(0, BM)
    nl = tl.arange(0, 2 * BN)
    am = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
    wn = tl.max_contiguous(tl.multiple_of(nl, 2 * BN), 2 * BN)
    rk = tl.arange(0, BK)
    ap = XP + am[:, None] * BK + rk[None, :]
    bp = WB + pn * (2 * BN) * K + wn[None, :] * BK + rk[:, None]
    acc = tl.zeros((BM, 2 * BN), tl.float32)
    for _ in range(tl.cdiv(K, BK)):
        if EVEN_M:
            a = tl.load(ap)
        else:
            a = tl.load(ap, mask=rm[:, None] < M, other=0.0)
        b = tl.load(bp)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        ap += M * BK
        bp += (2 * BN) * BK
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    gate, up = tl.split(tl.reshape(rounded, (BM, BN, 2)))
    # Equivalent real arithmetic is not necessarily identical FP32 arithmetic.
    result = gate * tl.sigmoid(gate) * up
    offset = pn * M * BN + rm[:, None] * BN + tl.arange(0, BN)[None, :]
    if EVEN_M:
        tl.store(Z + offset, result.to(tl.bfloat16))
    else:
        tl.store(Z + offset, result.to(tl.bfloat16), mask=rm[:, None] < M)


@triton.jit
def _packed_lm_head_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K, stride_im, stride_ik, stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, PACK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    input_ptrs = (
        input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    weight_ptrs = weight_ptr + (
        (offs_n[None, :] // PACK_N) * (K // BLOCK_SIZE_K) * PACK_N * BLOCK_SIZE_K
        + (offs_n[None, :] % PACK_N) * BLOCK_SIZE_K + offs_k[:, None]
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        input_mask_m = offs_m < M
        input_mask_k = offs_k < (K - k * BLOCK_SIZE_K)
        a = tl.load(
            input_ptrs,
            mask=input_mask_m[:, None] & input_mask_k[None, :],
            other=0.0,
        )
        weight_mask_k = offs_k < (K - k * BLOCK_SIZE_K)
        weight_mask_n = offs_n < N
        b = tl.load(
            weight_ptrs,
            mask=weight_mask_k[:, None] & weight_mask_n[None, :],
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += PACK_N * BLOCK_SIZE_K
    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = (
        output_ptr + offs_om[:, None] * stride_om + offs_on[None, :] * stride_on
    )
    output_mask_m = offs_m < M
    output_mask_n = offs_n < N
    output = accumulator.to(output_ptr.dtype.element_ty)
    tl.store(
        output_ptrs, output,
        mask=output_mask_m[:, None] & output_mask_n[None, :],
    )


def _check_device(x):
    if device.vendor_name != "iluvatar" or not x.is_cuda:
        raise NotImplementedError("packed bundle compute supports Iluvatar CUDA-like devices only")


def _check_inference(*tensors):
    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    ):
        raise NotImplementedError("packed bundle APIs are inference-only")


def _check_writes(reads, writes):
    """Reject every mutable buffer overlapping a read or another mutable buffer."""
    for index, (name, tensor) in enumerate(writes):
        for other_name, other in (*reads, *writes[:index]):
            if _overlaps(tensor, other):
                raise ValueError(f"{name} must not overlap {other_name}")


def pack_linear_weight(weight, *, panel_n: int, block_k: int):
    """Copy frozen BF16 [N,K] weights to [N/panel,K/block,panel,block].

    CPU and CUDA-like tensors are accepted. The detached result always owns
    a new contiguous allocation, including when the permutation is trivial.
    Callers must repack after changing the source weight.
    """
    for name, value in (("panel_n", panel_n), ("block_k", block_k)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [N, K]")
    _check_tensor(weight, tuple(weight.shape), "weight", contiguous=True)
    if weight.device.type not in ("cpu", "cuda"):
        raise NotImplementedError("weight packing supports CPU and CUDA-like devices only")
    _check_inference(weight)
    n, k = weight.shape
    if n <= 0 or k <= 0 or n % panel_n or k % block_k:
        raise ValueError("weight dimensions must be positive multiples of panel_n and block_k")
    context = (
        torch_device_fn.device(weight.device) if weight.is_cuda else nullcontext()
    )
    with context:
        return (
            weight.detach()
            .reshape(n // panel_n, panel_n, k // block_k, block_k)
            .permute(0, 2, 1, 3)
            .clone(memory_format=torch.contiguous_format)
        )


def packed_projection(x, packed_weight, *, out=None):
    """Compute BF16 [M,N] for positive M, N=2560 or 2048, using packed weights."""
    _check_device(x)
    m = _input_rows(x, D, contiguous=True)
    if packed_weight.ndim != 4:
        raise ValueError("packed weight must have shape [N/128, 64, 128, 32]")
    n = packed_weight.shape[0] * 128
    if n not in (2560, 2048):
        raise ValueError("packed projection supports N=2560 or N=2048 only")
    _check_tensor(
        packed_weight, (n // 128, 64, 128, 32), "packed weight",
        contiguous=True, device=x.device,
    )
    if out is not None:
        _check_tensor(out, (m, n), "out", contiguous=True, device=x.device)
        _check_writes((("x", x), ("packed weight", packed_weight)), (("out", out),))
    _check_inference(x, packed_weight, out)
    with torch_device_fn.device(x.device):
        if out is None:
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        _packed_nt_kernel[(triton.cdiv(m, 128) * (n // 128),)](
            x, packed_weight, out, m, n, D,
            BM=128, BN=128, BK=32, GROUP_M=8, PACK_A=False, EVEN_M=(m % 128 == 0),
            num_warps=8, num_stages=2,
        )
    return out


def packed_mlp(
    x, gate_packed, down_packed, *,
    input_workspace, activation_workspace, out=None,
):
    """Compute a positive-M/D2048/H6144 MLP using caller-owned K32 input/K64 Z.

    After the input copy, gate/up plus SiLU writes packed BF16 Z directly;
    a separate down GEMM consumes Z and packed down weights. The projection
    rounds to BF16 before FP32 activation and the BF16 Z boundary.
    """
    _check_device(x)
    m = _input_rows(x, D, contiguous=True)
    _check_tensor(
        gate_packed, WEIGHT_PACKED_SHAPE, "gate packed weight",
        contiguous=True, device=x.device,
    )
    _check_tensor(
        down_packed, DOWN_PACKED_SHAPE, "down packed weight",
        contiguous=True, device=x.device,
    )
    _check_tensor(
        input_workspace, (D // 32, m, 32), "input workspace",
        contiguous=True, device=x.device,
    )
    _check_tensor(
        activation_workspace, (H // 64, m, 64), "activation workspace",
        contiguous=True, device=x.device,
    )
    writes = (
        ("input workspace", input_workspace),
        ("activation workspace", activation_workspace),
    )
    if out is not None:
        _check_tensor(out, (m, D), "out", contiguous=True, device=x.device)
        writes += (("out", out),)
    _check_inference(
        x, gate_packed, down_packed, input_workspace, activation_workspace, out,
    )
    _check_writes(
        (("x", x), ("gate packed weight", gate_packed), ("down packed weight", down_packed)),
        writes,
    )
    with torch_device_fn.device(x.device):
        if out is None:
            out = torch.empty((m, D), dtype=x.dtype, device=x.device)
        xp = pack_gate_up_input(x, out=input_workspace)
        _packed_gate_up_silu_kernel[(triton.cdiv(m, 128) * (H // 64),)](
            xp, gate_packed, activation_workspace, m, H, D,
            BM=128, BN=64, BK=32, GROUP_M=8, EVEN_M=(m % 128 == 0),
            num_warps=8, num_stages=2,
        )
        _packed_nt_kernel[(triton.cdiv(m, 64) * (D // 64),)](
            activation_workspace, down_packed, out, m, D, H,
            BM=64, BN=64, BK=64, GROUP_M=8, PACK_A=True, EVEN_M=(m % 64 == 0),
            num_warps=4, num_stages=4,
        )
    return out


def packed_lm_head(x, packed_weight, *, out=None):
    """Compute full-vocabulary BF16 logits for M in (1,8,32,64), K2048."""
    _check_device(x)
    if x.ndim != 2 or x.shape[0] not in (1, 8, 32, 64):
        raise ValueError("packed LM head supports 2D input with M=1,8,32,64 only")
    m = x.shape[0]
    _check_tensor(x, (m, D), "x", contiguous=True)
    _check_tensor(
        packed_weight, LM_HEAD_PACKED_SHAPE, "packed weight",
        contiguous=True, device=x.device,
    )
    if out is not None:
        _check_tensor(out, (m, LM_HEAD_N), "out", contiguous=True, device=x.device)
        _check_writes((("x", x), ("packed weight", packed_weight)), (("out", out),))
    _check_inference(x, packed_weight, out)
    bm, bn, warps, stages = (32, 64, 2, 5) if m <= 32 else (64, 128, 4, 4)
    grid = (triton.cdiv(m, bm), triton.cdiv(LM_HEAD_N, bn))
    with torch_device_fn.device(x.device):
        if out is None:
            out = torch.empty((m, LM_HEAD_N), dtype=x.dtype, device=x.device)
        _packed_lm_head_kernel[grid](
            x, packed_weight, out, m, LM_HEAD_N, D,
            x.stride(0), x.stride(1), out.stride(0), out.stride(1),
            BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=32, PACK_N=128,
            num_warps=warps, num_stages=stages,
        )
    return out


__all__ = [
    "pack_linear_weight",
    "packed_projection",
    "packed_mlp",
    "packed_lm_head",
]
