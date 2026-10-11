# SPDX-License-Identifier: Apache-2.0
"""Integer-only fusion of the existing scheduler metadata arithmetic."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['num_tokens','num_reqs','max_reqs'])
def _fill(computed, req_indices, query_pos, scheduled, positions, seq_lens,
          num_tokens, num_reqs, max_reqs, block:tl.constexpr):
    off=tl.program_id(0)*block+tl.arange(0,block)
    req=tl.load(req_indices+off,off<num_tokens,other=0)
    value=tl.load(computed+req,off<num_tokens,other=0).to(tl.int64)
    query=tl.load(query_pos+off,off<num_tokens,other=0).to(tl.int64)
    tl.store(positions+off,value+query,off<num_tokens)
    count=tl.load(computed+off,off<num_reqs,other=0)
    added=tl.load(scheduled+off,off<num_reqs,other=0)
    length=tl.where(off<num_reqs,count+added,0)
    tl.store(seq_lens+off,length,off<max_reqs)


def fill(computed,req_indices,query_pos,scheduled,positions,seq_lens,num_tokens,num_reqs):
    assert computed.dtype==scheduled.dtype==seq_lens.dtype==torch.int32
    assert req_indices.dtype==query_pos.dtype==positions.dtype==torch.int64
    assert all(t.ndim==1 and t.is_cuda and t.is_contiguous() and t.device==computed.device
               for t in (computed,req_indices,query_pos,scheduled,positions,seq_lens))
    assert 0<=num_reqs<=computed.numel() and num_reqs<=scheduled.numel()
    assert num_tokens<=min(req_indices.numel(),query_pos.numel(),positions.numel())
    assert seq_lens.numel()>=computed.numel()
    extent=max(num_tokens,computed.numel())
    if extent:
        _fill[(triton.cdiv(extent,256),)](computed,req_indices,query_pos,scheduled,positions,
                                        seq_lens,num_tokens,num_reqs,computed.numel(),256,num_warps=4)
