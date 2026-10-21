# SPDX-License-Identifier: Apache-2.0
"""Experimental inference-only packed gate/up GEMM + SiLU on MetaX.

Supported contract: BF16 X[M,K2048] with positive M, weights[96,64,128,32],
and output[M,H6144]. No bias, quantization, gradients, or implicit
fallback. Unsupported model shapes must be handled by the framework.
Weight packing is a caller-managed, one-time operation for frozen weights.
"""

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

# Legacy M2048 allocation constants; computation derives rows from the input.
M, K, H = 2048, 2048, 6144
INPUT_PACKED_SHAPE = (K // 32, M, 32)
WEIGHT_PACKED_SHAPE = (H // 64, K // 32, 128, 32)


@triton.jit
def packed_gate_up_silu_kernel(XP,WB,Y,M,H,K,
                              BM:tl.constexpr,BN:tl.constexpr,
                              BK:tl.constexpr,GROUP_M:tl.constexpr,
                              EVEN_M:tl.constexpr=False):
    pid=tl.program_id(0)
    gm=tl.cdiv(M,BM)
    gn=tl.cdiv(H,BN)
    group=pid//(GROUP_M*gn)
    size=tl.minimum(gm-group*GROUP_M,GROUP_M)
    pm=group*GROUP_M+pid%size
    pn=(pid%(GROUP_M*gn))//size
    rm=pm*BM+tl.arange(0,BM)
    nl=tl.arange(0,2*BN)
    am=tl.max_contiguous(tl.multiple_of(rm,BM),BM)
    wn=tl.max_contiguous(tl.multiple_of(nl,2*BN),2*BN)
    rk=tl.arange(0,BK)
    ap=XP+am[:,None]*BK+rk[None,:]
    bp=WB+pn*(2*BN)*K+wn[None,:]*BK+rk[:,None]
    acc=tl.zeros((BM,2*BN),tl.float32)
    for _ in range(tl.cdiv(K,BK)):
        if EVEN_M:
            a=tl.load(ap)
        else:
            a=tl.load(ap,mask=rm[:,None]<M,other=0.0)
        b=tl.load(bp)
        acc=tl.dot(a,b,acc,out_dtype=tl.float32,allow_tf32=False)
        ap+=M*BK
        bp+=(2*BN)*BK
    rounded=acc.to(tl.bfloat16).to(tl.float32)
    gate,up=tl.split(tl.reshape(rounded,(BM,BN,2)))
    # Match Inductor-native FP32 operation order before the BF16 boundary.
    result=gate*tl.sigmoid(gate)*up
    outn=pn*BN+tl.arange(0,BN)
    if EVEN_M:
        tl.store(Y+rm[:,None]*H+outn[None,:],result.to(tl.bfloat16))
    else:
        tl.store(Y+rm[:,None]*H+outn[None,:],result.to(tl.bfloat16),mask=rm[:,None]<M)


def _check_tensor(tensor, shape, name, *, contiguous=False, device=None):
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must have dtype torch.bfloat16")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _input_rows(x, channels, *, contiguous=False):
    """Validate the fixed channel contract without fixing the token count."""
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != channels:
        raise ValueError(f"x must have shape [positive M, {channels}]")
    m = x.shape[0]
    _check_tensor(x, (m, channels), "x", contiguous=contiguous)
    return m


def _overlaps(a, b):
    """Conservative byte-span check; no device synchronization or data read."""
    if a.device != b.device:
        return False
    def span(tensor):
        extent = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride())
        )
        start = tensor.data_ptr()
        return start, start + extent * tensor.element_size()
    a_start, a_end = span(a)
    b_start, b_end = span(b)
    return a_start < b_end and b_start < a_end


def pack_gate_up_weight(weight):
    """Copy frozen [gate;up] BF16 weights to [Nblock,Kblock,pair,K32].

    Accepts CPU or device tensors. It does not cache weights or preserve
    gradients. The caller owns cache invalidation and graph lifetimes.
    """
    _check_tensor(weight, (2 * H, K), "weight", contiguous=True)
    return (
        weight.detach()
        .reshape(2, H // 64, 64, K // 32, 32)
        .permute(1, 3, 2, 0, 4)
        .contiguous()
        .reshape(WEIGHT_PACKED_SHAPE)
    )


def pack_gate_up_input(x, *, out=None):
    """Copy X to K32-major layout, optionally into a caller-owned workspace."""
    m = _input_rows(x, K)
    packed_shape = (K // 32, m, 32)
    if x.stride(1) != 1:
        raise ValueError("x must have unit stride in its K dimension")
    if out is None:
        out = torch.empty(packed_shape, dtype=x.dtype, device=x.device)
    _check_tensor(
        out, packed_shape, "input workspace",
        contiguous=True, device=x.device,
    )
    if _overlaps(x, out):
        raise ValueError("input workspace must not overlap x")
    out.copy_(x.reshape(m, K // 32, 32).permute(1, 0, 2))
    return out


def packed_gate_up_silu(x, packed_weight, *, input_workspace=None, out=None):
    """Compute BF16 SiLU(X @ W_gate.T) * (X @ W_up.T).

    Projection accumulates in FP32, rounds to BF16, and then performs the
    activation/product in FP32 before the final BF16 store. This API owns
    numerical computation and input layout conversion, not model integration,
    weight caches, stream-private workspace caches, or framework dispatch.
    It can be called eagerly or from an opaque custom op/CUDA graph after
    warmup. Performance on different M must be measured; other channels, dtypes, vendors, or
    training workloads is not claimed.
    """
    if runtime.device.vendor_name != "iluvatar" or not x.is_cuda:
        raise NotImplementedError("packed_gate_up_silu currently supports Iluvatar CUDA-like devices only")
    m = _input_rows(x, K)
    if x.stride(1) != 1:
        raise ValueError("x must have unit stride in its K dimension")
    _check_tensor(
        packed_weight, WEIGHT_PACKED_SHAPE, "packed weight",
        contiguous=True, device=x.device,
    )
    if torch.is_grad_enabled() and (x.requires_grad or packed_weight.requires_grad):
        raise NotImplementedError("packed_gate_up_silu is inference-only")
    if input_workspace is not None:
        _check_tensor(
            input_workspace, (K // 32, m, 32), "input workspace",
            contiguous=True, device=x.device,
        )
        if _overlaps(input_workspace, packed_weight):
            raise ValueError("input workspace must not overlap packed weights")
    if out is not None:
        _check_tensor(out, (m, H), "out", contiguous=True, device=x.device)
        if _overlaps(out, x) or _overlaps(out, packed_weight):
            raise ValueError("out must not overlap input or packed weights")
        if input_workspace is not None and _overlaps(out, input_workspace):
            raise ValueError("out must not overlap input workspace")
    with torch_device_fn.device(x.device):
        xp = pack_gate_up_input(x, out=input_workspace)
        if out is None:
            out = torch.empty((m, H), dtype=x.dtype, device=x.device)
        packed_gate_up_silu_kernel[(triton.cdiv(m, 128) * (H // 64),)](
            xp, packed_weight, out, m, H, K,
            BM=128, BN=64, BK=32, GROUP_M=8, EVEN_M=(m % 128 == 0),
            num_warps=8, num_stages=2,
        )
    return out


__all__ = ["pack_gate_up_weight", "pack_gate_up_input", "packed_gate_up_silu"]
