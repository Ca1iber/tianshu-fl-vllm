# SPDX-License-Identifier: Apache-2.0
"""Candidate small-batch mask routing; no sampler/RNG replacement."""
import hashlib
from pathlib import Path

SAMPLER_SHA = '35b9a1ffeb1e50b73b0ddd1ffc0f968910ee21cf9cc5455dd879fde547b6ddc1'
GEO_SHA = '09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'


def build_route(original, geo, torch, observer=None):
    """Build the candidate for already validated vLLM top-p input tensors.

    Upstream vLLM validates probability values. This routing decision reads
    metadata only; it must not synchronize to inspect device tensor values.
    The candidate still requires broader-p GPU checks and fresh model quality.
    """
    def apply(logits, k, p):
        supported = (
            k is None and p is not None
            and logits.ndim == 2 and 1 <= logits.shape[0] <= 7
            and logits.shape[1] == 130560 and logits.dtype == torch.float32
            and logits.device.type == 'cuda' and logits.is_contiguous()
            and p.ndim == 1 and p.shape[0] == logits.shape[0]
            and p.dtype == torch.float32 and p.device == logits.device
            and p.is_contiguous()
        )
        if not supported:
            return original(logits, k, p)
        result = geo.apply_top_k_top_p_triton(logits, k, p)
        if observer is not None:
            observer(logits)
        return result
    return apply


def install(sampler, geo, torch, observer=None, compiled_observer=None):
    """Install before any consumers copy the module's mask-function alias."""
    assert hashlib.sha256(Path(sampler.__file__).read_bytes()).hexdigest() == SAMPLER_SHA
    assert hashlib.sha256(Path(geo.__file__).read_bytes()).hexdigest() == GEO_SHA
    assert not getattr(sampler.apply_top_k_top_p, '_tianshu_tiny_candidate', False)
    original = sampler.apply_top_k_top_p
    kernel = geo._topk_topp_kernel

    class KernelRoute:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                if (1 <= kwargs.get('BATCH_SIZE', 0) <= 7
                        and kwargs.get('VOCAB_SIZE') == 130560
                        and kwargs.get('TOPP_ENABLED') and not kwargs.get('TOPK_ENABLED')):
                    kwargs.update(BLOCK_SIZE=32768, BLOCK_SIZE_TRUNC=16384,
                                  num_warps=16, num_stages=1)
                result = kernel[grid](*args, **kwargs)
                if compiled_observer is not None:
                    compiled_observer(kwargs, result)
                return result
            return launch

    geo._topk_topp_kernel = KernelRoute()
    routed = build_route(original, geo, torch, observer=observer)
    routed._tianshu_tiny_candidate = True
    sampler.apply_top_k_top_p = routed
    sampler._tianshu_tiny_original = original
    return dict(original=original, original_geo_kernel=kernel, routed=routed,
                scope='Mask only; no RNG, sampling mode, temperature or p mutation',
                adopted=False)
