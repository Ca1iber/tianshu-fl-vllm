# SPDX-License-Identifier: Apache-2.0
"""Tianshu IR RMS provider preserving native FP32 accumulation and weight rounding."""
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import Tensor
import triton
import triton.language as tl

PROVIDER = "tianshu_ir_rms"
_SEEN = set()


@triton.jit(do_not_specialize=["epsilon"])
def rms_kernel(x, weight, output, epsilon, N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, N)
    values = tl.load(x + row * N + cols).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / N
    normalized = (values * tl.rsqrt(variance + epsilon)).to(weight.dtype.element_ty)
    weighted = normalized * tl.load(weight + cols)
    tl.store(output + row * N + cols, weighted)


@triton.jit(do_not_specialize=["epsilon"])
def fused_kernel(x, x_residual, weight, output, residual_output, epsilon, N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, N)
    values = tl.load(x + row * N + cols).to(tl.float32)
    values += tl.load(x_residual + row * N + cols).to(tl.float32)
    tl.store(residual_output + row * N + cols, values)
    variance = tl.sum(values * values, axis=0) / N
    normalized = (values * tl.rsqrt(variance + epsilon)).to(weight.dtype.element_ty)
    weighted = normalized * tl.load(weight + cols)
    tl.store(output + row * N + cols, weighted)


def supports_rms(x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None) -> bool:
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 2
            and x.shape[1] == 2048 and 1 <= x.shape[0] <= 2048 and x.is_contiguous()
            and weight is not None and weight.is_cuda and weight.is_contiguous()
            and weight.shape == (2048,) and weight.dtype in (torch.bfloat16, torch.float32)
            and variance_size in (None, 2048))


def supports_fused(x: Tensor, x_residual: Tensor, weight: Tensor | None,
                   epsilon: float, variance_size: int | None = None) -> bool:
    return (supports_rms(x, weight, epsilon, variance_size)
            and x_residual.shape == x.shape and x_residual.dtype == x.dtype
            and x_residual.is_cuda and x_residual.is_contiguous())


def record(kind, x, weight, compiled):
    directory = os.environ.get("TIANSHU_IR_RMS_RECORD_DIR")
    key = (kind, int(x.shape[0]), str(weight.dtype))
    if not directory or key in _SEEN:
        return
    _SEEN.add(key)
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    data = dict(provider=PROVIDER, kind=kind, rows=key[1], columns=2048,
                activation_dtype=str(x.dtype), weight_dtype=key[2],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                functional_inputs_unchanged=True, native_normalized_weight_dtype_rounding_preserved=True,
                compiled_metadata={k: getattr(compiled.metadata, k, None)
                                   for k in ("num_warps", "num_stages", "shared")})
    (root / f"{kind}_{os.getpid()}_{len(_SEEN)}.json").write_text(json.dumps(data, indent=2) + "\n")


def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float,
             variance_size: int | None = None) -> Tensor:
    assert weight is not None
    output = torch.empty_like(x)
    warps = 8 if x.shape[0] <= 64 else 4
    compiled = rms_kernel[(x.shape[0],)](x, weight, output, epsilon, 2048, num_warps=warps, num_stages=1)
    record("normal", x, weight, compiled)
    return output


def fused_add_rms_norm(x: Tensor, x_residual: Tensor, weight: Tensor | None,
                       epsilon: float, variance_size: int | None = None) -> tuple[Tensor, Tensor]:
    assert weight is not None
    output = torch.empty_like(x)
    residual_output = torch.empty_like(x_residual)
    warps = 8 if x.shape[0] <= 64 else 4
    compiled = fused_kernel[(x.shape[0],)](x, x_residual, weight, output, residual_output,
                                        epsilon, 2048, num_warps=warps, num_stages=1)
    record("fused", x, weight, compiled)
    return output, residual_output


def register(ir_module):
    if PROVIDER not in ir_module.rms_norm.impls:
        ir_module.rms_norm.register_impl(PROVIDER, supported=torch.cuda.is_available(), supports_args=supports_rms)(rms_norm)
    if PROVIDER not in ir_module.fused_add_rms_norm.impls:
        ir_module.fused_add_rms_norm.register_impl(PROVIDER, supported=torch.cuda.is_available(), supports_args=supports_fused)(fused_add_rms_norm)
