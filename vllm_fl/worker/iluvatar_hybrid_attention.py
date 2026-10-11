# SPDX-License-Identifier: Apache-2.0
"""Opt-in MiniCPM routing: native 3D Decode, FlagGems FA2 Prefill.

KV writes and the native metadata remain owned by TritonAttentionImpl.
All tensor arithmetic lives in existing kernels. No quantization/sampling
changes. CPU query boundaries are inspected once when metadata is built.
"""
from contextvars import ContextVar
from dataclasses import dataclass
import functools
import os

import torch

_ROUTE = ContextVar('fl_iluvatar_hybrid_route', default=None)
_INSTALLED = False
COUNTS = {'native_decode': 0, 'prefill': 0, 'mixed': 0, 'fallback': 0}


@dataclass
class Route:
    num_decodes: int
    prefill_cu: torch.Tensor
    max_prefill_query: int


def route_from_common(common):
    starts = common.query_start_loc_cpu[:common.num_reqs+1]
    if starts.numel() != common.num_reqs+1 or starts[0].item() != 0 or starts[-1].item() != common.num_actual_tokens:
        return None
    lengths = (starts[1:]-starts[:-1]).tolist()
    if not lengths or any(x <= 0 for x in lengths):
        return None
    if all(x == 1 for x in lengths):
        return None
    decodes = 0
    for length in lengths:
        if length != 1:
            break
        decodes += 1
    if decodes > 64 or any(x <= 1 for x in lengths[decodes:]):
        return None
    # One GPU subtraction per model step; shared by all attention layers.
    cu = common.query_start_loc[decodes:common.num_reqs+1]
    if decodes:
        cu = cu - decodes
    return Route(decodes, cu, max(lengths[decodes:]))


def supported(kwargs):
    from vllm.v1.kv_cache_interface import KVQuantMode
    q, k, v, out = (kwargs[x] for x in ('q','k','v','out'))
    return (q.ndim == 3 and q.shape[1:] == (16,128)
        and k.ndim == 4 and k.shape[1:] == (16,2,128) and v.shape == k.shape
        and q.dtype == k.dtype == v.dtype == out.dtype == torch.bfloat16
        and q.is_cuda and q.device == k.device == v.device == out.device
        and kwargs.get('causal') is True and tuple(kwargs.get('window_size',())) == (-1,-1)
        and kwargs.get('softcap',0.) == 0. and kwargs.get('output_scale') is None
        and kwargs.get('kv_quant_mode',KVQuantMode.NONE) == KVQuantMode.NONE
        and all(kwargs.get(x) is None for x in ('alibi_slopes','qq_bias','sinks','mm_prefix_range'))
        and not kwargs.get('use_td',False) and kwargs.get('chunk_lookback',-1) == -1)


def dispatch(original, route, kwargs):
    from flag_gems import flash_attn_varlen_func
    if kwargs['max_seqlen_q'] <= 1:
        COUNTS['native_decode'] += 1
        return original(**kwargs)
    if route is None or not supported(kwargs):
        COUNTS['fallback'] += 1
        return original(**kwargs)
    nd = route.num_decodes
    if nd:
        # The Decode query starts already equal [0,1,...,nd]. Keep native
        # segments/buffers, matching the baseline's pure Decode path.
        decode = dict(kwargs)
        decode.update(q=kwargs['q'][:nd], out=kwargs['out'][:nd],
            cu_seqlens_q=kwargs['cu_seqlens_q'][:nd+1], max_seqlen_q=1,
            seqused_k=kwargs['seqused_k'][:nd], block_table=kwargs['block_table'][:nd])
        original(**decode)
        COUNTS['mixed'] += 1
    else:
        COUNTS['prefill'] += 1
    flash_attn_varlen_func(q=kwargs['q'][nd:], k=kwargs['k'], v=kwargs['v'],
        out=kwargs['out'][nd:], cu_seqlens_q=route.prefill_cu,
        max_seqlen_q=route.max_prefill_query, seqused_k=kwargs['seqused_k'][nd:],
        max_seqlen_k=kwargs['max_seqlen_k'], softmax_scale=kwargs['softmax_scale'],
        causal=True, window_size=(-1,-1), block_table=kwargs['block_table'][nd:],
        softcap=0., num_splits=0, fa_version=2)
    return None


def configure(config, logger):
    global _INSTALLED
    if os.getenv('VLLM_FL_ILUVATAR_HYBRID_ATTENTION','0') != '1':
        return False
    from vllm.platforms import current_platform
    text, parallel = config.model_config.hf_config, config.parallel_config
    if (getattr(current_platform,'vendor_name',None) != 'iluvatar'
        or config.model_config.dtype != torch.bfloat16
        or config.model_config.quantization is not None
        or config.lora_config is not None or config.speculative_config is not None
        or parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1 or parallel.decode_context_parallel_size != 1
        or getattr(text,'hidden_size',None) != 2048 or getattr(text,'num_hidden_layers',None) != 42
        or getattr(text,'num_attention_heads',None) != 16 or getattr(text,'num_key_value_heads',None) != 2
        or config.model_config.is_multimodal_model
        or os.getenv('VLLM_FL_USE_FLAGGEMS_ATTN','0') == '1'):
        logger.warning('Hybrid attention unsupported; keeping native routing')
        return False
    if _INSTALLED:
        return True
    import vllm.v1.attention.backends.triton_attn as backend
    import vllm.v1.attention.ops.triton_unified_attention as native
    if native.is_batch_invariant:
        logger.warning('Hybrid attention disabled for batch-invariant execution')
        return False
    builder = backend.TritonAttentionMetadataBuilder
    original_build = builder.build
    original_forward = backend.TritonAttentionImpl.forward
    original_unified = backend.unified_attention
    builder.reorder_batch_threshold = 1

    @functools.wraps(original_build)
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = original_build(self,common_prefix_len,common_attn_metadata,fast_build=fast_build)
        result._fl_hybrid_route = (route_from_common(common_attn_metadata)
            if not result.use_cascade and result.max_query_len > 1 else None)
        return result

    @functools.wraps(original_forward)
    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output,
                output_scale=None,output_block_scale=None):
        route = getattr(attn_metadata,'_fl_hybrid_route',None)
        if route is None:
            return original_forward(self,layer,query,key,value,kv_cache,attn_metadata,output,
                output_scale=output_scale,output_block_scale=output_block_scale)
        token = _ROUTE.set(route)
        try:
            return original_forward(self,layer,query,key,value,kv_cache,attn_metadata,output,
                output_scale=output_scale,output_block_scale=output_block_scale)
        finally:
            _ROUTE.reset(token)

    @functools.wraps(original_unified)
    def unified(*args,**kwargs):
        if args:
            return original_unified(*args,**kwargs)
        return dispatch(original_unified,_ROUTE.get(),kwargs)

    builder.build = build
    backend.TritonAttentionImpl.forward = forward
    backend.unified_attention = unified
    _INSTALLED = True
    logger.warning('Enabled Iluvatar hybrid attention: native Decode, FlagGems FA2 Prefill')
    return True
