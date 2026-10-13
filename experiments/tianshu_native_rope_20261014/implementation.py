# SPDX-License-Identifier: Apache-2.0
"""Full Neox RoPE Q/K fusion with native BF16 intermediate rounding."""
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import Tensor
import triton
import triton.language as tl

_SEEN = set()


@triton.jit
def rope_kernel(positions, query, key, cache, out_query, out_key,
                Q_ROW: tl.constexpr, K_ROW: tl.constexpr, CACHE_ROW: tl.constexpr,
                HALF: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    i = tl.arange(0, HALF)
    # Keep the actual int64 position for cache pointer arithmetic.
    position = tl.load(positions + row)
    cosine = tl.load(cache + position * CACHE_ROW + i).to(tl.float32)
    sine = tl.load(cache + position * CACHE_ROW + HALF + i).to(tl.float32)
    if head < 16:
        base = row * Q_ROW + head * 128
        x1 = tl.load(query + base + i).to(tl.float32)
        x2 = tl.load(query + base + HALF + i).to(tl.float32)
    else:
        base = row * K_ROW + (head - 16) * 128
        x1 = tl.load(key + base + i).to(tl.float32)
        x2 = tl.load(key + base + HALF + i).to(tl.float32)
    # PyTorch native performs these four products in BF16 before add/sub.
    p1 = (x1 * cosine).to(tl.bfloat16).to(tl.float32)
    p2 = (x2 * sine).to(tl.bfloat16).to(tl.float32)
    p3 = (x2 * cosine).to(tl.bfloat16).to(tl.float32)
    p4 = (x1 * sine).to(tl.bfloat16).to(tl.float32)
    first = p1 - p2
    second = p3 + p4
    if head < 16:
        out_base = row * 2048 + head * 128
        tl.store(out_query + out_base + i, first)
        tl.store(out_query + out_base + HALF + i, second)
    else:
        out_base = row * 256 + (head - 16) * 128
        tl.store(out_key + out_base + i, first)
        tl.store(out_key + out_base + HALF + i, second)


@torch.library.custom_op("tianshu_native_rope::qk", mutates_args=(), device_types="cuda")
def opaque_rope(positions: Tensor, query: Tensor, key: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
    out_query = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    out_key = torch.empty(key.shape, device=key.device, dtype=key.dtype)
    compiled = rope_kernel[(query.shape[0], 18)](positions, query, key, cache, out_query, out_key,
                                                 query.stride(0), key.stride(0), cache.stride(0), 64,
                                                 num_warps=4, num_stages=1, enable_fp_fusion=False)
    directory = os.environ.get("TIANSHU_NATIVE_ROPE_RECORD_DIR")
    ident = (int(query.shape[0]), int(query.stride(0)), int(key.stride(0)))
    if directory and ident not in _SEEN:
        _SEEN.add(ident); root = Path(directory); root.mkdir(parents=True, exist_ok=True)
        data = dict(rows=ident[0], query_row_stride=ident[1], key_row_stride=ident[2],
                    query_heads=16, key_heads=2, head_size=128, neox=True,
                    position_dtype=str(positions.dtype), intermediate_product_dtype="bfloat16",
                    fp_fusion_disabled=True, inputs_unchanged=True,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    compiled_metadata={k: getattr(compiled.metadata, k, None)
                                       for k in ("num_warps", "num_stages", "shared")})
        (root / f"rope_{os.getpid()}_{len(_SEEN)}.json").write_text(json.dumps(data, indent=2) + "\n")
    return out_query, out_key


@opaque_rope.register_fake
def fake_rope(positions, query, key, cache):
    return torch.empty(query.shape, device=query.device, dtype=query.dtype), torch.empty(key.shape, device=key.device, dtype=key.dtype)


def supports(positions, query, key, head_size, rotary_dim, cache, neox):
    return (key is not None and query.is_cuda and key.is_cuda and positions.is_cuda and cache.is_cuda
            and query.dtype == key.dtype == cache.dtype == torch.bfloat16
            and positions.dtype == torch.int64 and positions.is_contiguous()
            and positions.ndim == 1 and query.ndim == key.ndim == cache.ndim == 2
            and 1 <= query.shape[0] <= 2048 and key.shape[0] == query.shape[0] == positions.shape[0]
            and query.shape[1] == 2048 and key.shape[1] == 256 and cache.shape[1] == 128
            and query.stride(1) == key.stride(1) == cache.stride(1) == 1
            and head_size == rotary_dim == 128 and neox)


def install(module):
    original = module.RotaryEmbedding.forward_static

    def forward_static(positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox_style):
        if supports(positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox_style):
            return opaque_rope(positions, query, key, cos_sin_cache)
        return original(positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox_style)

    module.RotaryEmbedding._tianshu_original_forward_static = staticmethod(original)
    module.RotaryEmbedding.forward_static = staticmethod(forward_static)
