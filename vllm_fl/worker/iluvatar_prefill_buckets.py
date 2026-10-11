# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in prefill GEMM buckets, retaining FULL_DECODE_ONLY attention dispatch."""
from bisect import bisect_left
import os


PREFILL_BUCKETS = tuple(sorted(set(
    [1, 2, 4]
    + list(range(8, 257, 8))
    + list(range(272, 513, 16))
    + list(range(544, 1025, 32))
    + list(range(1152, 2049, 128))
)))


def buckets_for_budget(max_tokens: int) -> tuple[int, ...]:
    """A bounded ceiling map; never grow runner buffers or scheduler budget."""
    if max_tokens <= 0 or max_tokens > 2048:
        raise ValueError('Bucket policy supports token budgets in 1..2048')
    return tuple(sorted(set(x for x in PREFILL_BUCKETS if x <= max_tokens) | {max_tokens}))


def padded_tokens(num_tokens: int, max_tokens: int) -> int:
    if not 1 <= num_tokens <= max_tokens:
        raise ValueError('Token count must fit the runner budget')
    sizes = buckets_for_budget(max_tokens)
    return sizes[bisect_left(sizes, num_tokens)]


def enabled(vllm_config, platform) -> bool:
    """Initially restricted to the competition's dense MiniCPM geometry."""
    if os.environ.get('VLLM_FL_ILUVATAR_PREFILL_BUCKETS', '0') != '1':
        return False
    if getattr(platform, 'vendor_name', None) != 'iluvatar':
        return False
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    hf = getattr(model, 'hf_text_config', None)
    scheduler = vllm_config.scheduler_config
    return bool(
        model is not None and hf is not None
        and getattr(hf, 'model_type', None) == 'llama'
        and getattr(hf, 'hidden_size', None) == 2048
        and getattr(hf, 'intermediate_size', None) == 6144
        and getattr(hf, 'num_hidden_layers', None) == 42
        and getattr(hf, 'num_attention_heads', None) == 16
        and getattr(hf, 'num_key_value_heads', None) == 2
        and getattr(model, 'quantization', None) is None
        and not getattr(model, 'is_multimodal_model', False)
        and not getattr(model, 'is_encoder_decoder', False)
        and parallel.tensor_parallel_size == parallel.pipeline_parallel_size == parallel.data_parallel_size == 1
        and not getattr(parallel, 'use_ubatching', False)
        and vllm_config.lora_config is None
        and vllm_config.speculative_config is None
        and 1 <= scheduler.max_num_batched_tokens <= 2048
        and vllm_config.compilation_config.cudagraph_mode.name == 'FULL_DECODE_ONLY'
    )


def configure(vllm_config, platform, logger) -> None:
    if not enabled(vllm_config, platform):
        return
    sizes = buckets_for_budget(vllm_config.scheduler_config.max_num_batched_tokens)
    cfg = vllm_config.compilation_config
    cfg.cudagraph_capture_sizes = list(sizes)
    cfg.max_cudagraph_capture_size = sizes[-1]
    logger.info('Iluvatar prefill buckets enabled: %d points, max=%d; retaining FULL_DECODE_ONLY',
                len(sizes), sizes[-1])


def warmup(worker, logger) -> None:
    """Warm mixed/prefill rows even though decode-only graphs exclude them."""
    from vllm.config import CUDAGraphMode
    from vllm.platforms import current_platform

    if not enabled(worker.vllm_config, current_platform):
        return
    sizes = buckets_for_budget(worker.scheduler_config.max_num_batched_tokens)
    gather_warmed = False
    for size in reversed(sizes):
        logger.info('Iluvatar prefill bucket warmup M=%d', size)
        hidden, _ = worker.model_runner._dummy_run(
            size, skip_eplb=True, remove_lora=False,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        if not gather_warmed:
            import torch
            from .iluvatar_logits_gather import enabled as gather_enabled, gather_rows
            if gather_enabled(worker.vllm_config, current_platform):
                # Dummy selection uses int64; runtime scheduler supplies int32.
                indices = torch.arange(min(size, 64), device=hidden.device, dtype=torch.int32)
                gather_rows(hidden, indices)
            gather_warmed = True
    logger.info('Iluvatar prefill bucket warmup complete: %d points', len(sizes))
