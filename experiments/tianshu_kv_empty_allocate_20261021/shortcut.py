# SPDX-License-Identifier: Apache-2.0
"""Full-attention decode: return native empty object only when all native stages do no work."""
import functools

def wrap(manager,original,RequestType,running_status,mode):
    assert mode in ('control','candidate')
    assert len(manager.coordinator.single_type_managers)==1
    unit=manager.coordinator.single_type_managers[0]
    assert type(unit).__name__=='FullAttentionManager' and type(unit.kv_cache_spec).__name__=='FullAttentionSpec'
    assert unit.block_size==16 and not unit.use_eagle and manager.enable_caching and not manager.use_eagle
    assert manager.max_model_len==131072 and manager.watermark_blocks==0 and manager.metrics_collector is None
    stats=dict(calls=0,eligible=0,shortcuts=0,native_calls=0)
    @functools.wraps(original)
    def allocate(request,num_new_tokens,num_new_computed_tokens=0,new_computed_blocks=None,num_lookahead_tokens=0,
                 num_external_computed_tokens=0,delay_cache_blocks=False,num_encoder_tokens=0,
                 full_sequence_must_fit=False,reserved_blocks=0,has_scheduled_reqs=True):
        stats['calls']+=1
        eligible=False
        if (type(request) is RequestType and request.status==running_status and type(num_new_tokens) is int and num_new_tokens==1
            and num_new_computed_tokens==0 and new_computed_blocks is None and num_lookahead_tokens==0
            and num_external_computed_tokens==0 and delay_cache_blocks is False and num_encoder_tokens==0
            and full_sequence_must_fit is False and reserved_blocks==0):
            computed=request.num_computed_tokens
            if type(computed) is int and request.num_prompt_tokens<=computed<manager.max_model_len:
                blocks=unit.req_to_blocks.get(request.request_id)
                cached=unit.num_cached_block.get(request.request_id)
                end=computed+1
                eligible=(blocks is not None and cached is not None and len(blocks)*16>=end
                    and cached>=min(end,request.num_tokens)//16)
        if eligible:
            stats['eligible']+=1
            if mode=='candidate':
                stats['shortcuts']+=1
                return manager.empty_kv_cache_blocks
        stats['native_calls']+=1
        return original(request,num_new_tokens,num_new_computed_tokens,new_computed_blocks,num_lookahead_tokens,
            num_external_computed_tokens,delay_cache_blocks,num_encoder_tokens,full_sequence_must_fit,reserved_blocks,has_scheduled_reqs)
    return allocate,stats
