# SPDX-License-Identifier: Apache-2.0
"""Opt-in native Decode segmentation using one reusable S64 scratch arena."""
import functools
import os

import torch

_INSTALLED = False
COUNTS = {4: 0, 16: 0, 64: 0}


def choose_segments(num_sequences):
    if num_sequences == 1:
        return 64
    if 16 <= num_sequences <= 64:
        return 4
    return 16


def configure(config, logger):
    global _INSTALLED
    if os.getenv('VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS', '0') != '1':
        return False
    from vllm.platforms import current_platform
    text, parallel = config.model_config.hf_config, config.parallel_config
    if (getattr(current_platform, 'vendor_name', None) != 'iluvatar'
        or config.model_config.dtype != torch.bfloat16
        or config.model_config.quantization is not None
        or config.lora_config is not None or config.speculative_config is not None
        or parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1 or parallel.decode_context_parallel_size != 1
        or getattr(text, 'hidden_size', None) != 2048
        or getattr(text, 'num_hidden_layers', None) != 42
        or getattr(text, 'num_attention_heads', None) != 16
        or getattr(text, 'num_key_value_heads', None) != 2
        or config.model_config.is_multimodal_model
        or os.getenv('VLLM_FL_USE_FLAGGEMS_ATTN', '0') == '1'):
        logger.warning('Adaptive Decode unsupported; keeping native configuration')
        return False
    if _INSTALLED:
        return True
    import vllm.v1.attention.backends.triton_attn as backend
    import vllm.v1.attention.ops.triton_unified_attention as native
    if native.is_batch_invariant:
        return False
    original = backend.unified_attention
    # Builders are constructed after WorkerFL initialization. Both native
    # kernels use NUM_SEGMENTS as their packed stride, not tensor.stride().
    # A logical S4/S16 arena therefore occupies a prefix of the S64 storage.
    backend.NUM_PAR_SOFTMAX_SEGMENTS = 64

    @functools.wraps(original)
    def unified(*args, **kwargs):
        if args or kwargs.get('max_seqlen_q') != 1:
            return original(*args, **kwargs)
        q = kwargs['q']
        num_sequences = kwargs['cu_seqlens_q'].numel() - 1
        threshold = kwargs.get('seq_threshold_3D')
        scratch = [kwargs.get(name) for name in
                   ('softmax_segm_output', 'softmax_segm_max', 'softmax_segm_expsum')]
        if (q.dtype != torch.bfloat16 or q.shape[1:] != (16, 128)
            or threshold is None or not 1 <= num_sequences <= min(64, threshold)
            or any(t is None or not t.is_contiguous() or t.dtype != torch.float32
                   or t.device != q.device for t in scratch)):
            return original(**kwargs)
        segments = choose_segments(num_sequences)
        required = num_sequences * 16 * segments
        if scratch[0].numel() < required * 128 or any(t.numel() < required for t in scratch[1:]):
            return original(**kwargs)
        kwargs['num_par_softmax_segments'] = segments
        COUNTS[segments] += 1
        return original(**kwargs)

    backend.unified_attention = unified
    _INSTALLED = True
    logger.warning('Enabled adaptive native Decode: B1 S64, B2-15 S16, B16-64 S4; shared S64 scratch')
    return True
