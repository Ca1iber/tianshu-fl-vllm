# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed launch for the scheduler's valid, nonnegative last-token row indices."""
import os
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['source_rows', 'num_rows', 'stride_row', 'stride_col'])
def _gather_rows(source, indices, output, source_rows, num_rows, stride_row, stride_col,
                 WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    index = tl.load(indices + row)
    valid = (cols < WIDTH) & (index >= 0) & (index < source_rows)
    values = tl.load(source + index * stride_row + cols * stride_col, mask=valid, other=0)
    tl.store(output + row * WIDTH + cols, values, mask=cols < WIDTH)


def gather_rows(hidden_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 2 or hidden_states.shape[1] != 2048:
        raise ValueError('Initial gather implementation requires 2D hidden width 2048')
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64) or not indices.is_contiguous():
        raise ValueError('Expected contiguous scheduler row indices (int32/int64)')
    rows = indices.numel()
    output = torch.empty((rows, 2048), dtype=hidden_states.dtype, device=hidden_states.device)
    if rows:
        _gather_rows[(rows, 8)](hidden_states, indices, output, hidden_states.shape[0], rows,
                               hidden_states.stride(0), hidden_states.stride(1),
                               WIDTH=2048, BLOCK=256, num_warps=4)
    return output


def enabled(vllm_config, platform) -> bool:
    """Experimental geometry gate independent of the Prefill padding policy."""
    if os.environ.get('VLLM_FL_ILUVATAR_LOGITS_GATHER', '0') != '1':
        return False
    model, parallel = vllm_config.model_config, vllm_config.parallel_config
    hf = getattr(model, 'hf_text_config', None)
    scheduler = vllm_config.scheduler_config
    return bool(
        getattr(platform, 'vendor_name', None) == 'iluvatar'
        and model is not None and hf is not None
        and getattr(hf, 'model_type', None) == 'llama'
        and getattr(hf, 'hidden_size', None) == 2048
        and getattr(hf, 'intermediate_size', None) == 6144
        and getattr(hf, 'num_hidden_layers', None) == 42
        and getattr(hf, 'num_attention_heads', None) == 16
        and getattr(hf, 'num_key_value_heads', None) == 2
        and model.dtype == torch.bfloat16
        and model.quantization is None
        and not getattr(model, 'is_multimodal_model', False)
        and not getattr(model, 'is_encoder_decoder', False)
        and parallel.tensor_parallel_size == parallel.pipeline_parallel_size == parallel.data_parallel_size == 1
        and not getattr(parallel, 'use_ubatching', False)
        and vllm_config.lora_config is None and vllm_config.speculative_config is None
        and 1 <= scheduler.max_num_batched_tokens <= 8192
        and vllm_config.compilation_config.cudagraph_mode.name == 'FULL_DECODE_ONLY'
    )
