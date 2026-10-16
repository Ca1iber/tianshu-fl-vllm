# SPDX-License-Identifier: Apache-2.0
"""Fuse stable pure-decode GPU batch metadata; retain original integer dtypes."""
import torch
import triton
import triton.language as tl


def eligible(owner, scheduler_output, counts):
    batch=owner.input_batch
    n=batch.num_reqs
    if not 1<=n<=64 or len(counts)!=n or not bool((counts==1).all()):
        return False
    if (not owner.use_async_scheduling or owner.use_async_spec_decode
            or owner.speculative_config is not None or scheduler_output.scheduled_spec_decode_tokens
            or owner.enable_prompt_embeds or batch.req_prompt_embeds
            or owner.uses_mrope or owner.uses_xdrope_dim>0
            or owner.num_accepted_tokens_event is not None
            or owner.mamba_prev_last_scheduled_idx is not None):
        return False
    if scheduler_output.total_num_scheduled_tokens!=n:
        return False
    previous=batch.prev_sampled_token_ids
    if (previous is None or previous.device.type!='cuda' or previous.device!=owner.input_ids.gpu.device or previous.ndim!=2
            or previous.shape[0]<n or previous.shape[1]!=1):
        return False
    old=batch.prev_req_id_to_index
    for index,request_id in enumerate(batch.req_ids[:n]):
        if old.get(request_id,-1)!=index:
            return False
        computed=int(batch.num_computed_tokens_cpu[index])
        if (computed<int(batch.num_prompt_tokens[index])
                or computed+1<owner.requests[request_id].num_tokens
                or not 0<=computed<owner.max_model_len):
            return False
    return True


@triton.jit(do_not_specialize=['n'])
def _prepare_decode(computed, previous, input_ids, positions, query_start, seq_lens,
                    accepted, req_indices, query_pos, scheduled, discard, logits_indices,
                    n, previous_stride:tl.constexpr, query_size:tl.constexpr,
                    seq_size:tl.constexpr, accepted_size:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK)
    valid=i<n
    value=tl.load(computed+i,valid,other=0)
    token=tl.load(previous+i*previous_stride,valid,other=0)
    tl.store(input_ids+i,token,valid)
    tl.store(positions+i,value.to(tl.int64),valid)
    tl.store(query_start+i,tl.minimum(i,n),i<query_size)
    tl.store(seq_lens+i,tl.where(valid,value+1,0),i<seq_size)
    tl.store(accepted+i,1,i<accepted_size)
    tl.store(req_indices+i,i.to(tl.int64),valid)
    tl.store(query_pos+i,0,valid)
    tl.store(scheduled+i,1,valid)
    tl.store(discard+i,False,valid)
    tl.store(logits_indices+i,i,valid)


def launch(owner,n):
    if not hasattr(owner,'_tianshu_decode_logit_indices'):
        owner._tianshu_decode_logit_indices=torch.empty(64,dtype=torch.int32,device=owner.device)
    query=owner.query_start_loc.gpu
    accepted=owner.num_accepted_tokens.gpu
    previous=owner.input_batch.prev_sampled_token_ids
    block=triton.next_power_of_2(max(query.numel(),owner.seq_lens.numel(),accepted.numel()))
    _prepare_decode[(1,)](owner.num_computed_tokens,previous,owner.input_ids.gpu,owner.positions,
        query,owner.seq_lens,accepted,owner.req_indices.gpu,owner.query_pos.gpu,
        owner.num_scheduled_tokens.gpu,owner.discard_request_mask.gpu,owner._tianshu_decode_logit_indices,
        n,previous.stride(0),query.numel(),owner.seq_lens.numel(),accepted.numel(),BLOCK=block,num_warps=4,num_stages=1)
