# SPDX-License-Identifier: Apache-2.0
"""CPU metadata only. No deprecated length property or GPU transfer."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class PrefillBound:
    max_key_length:int
    num_prefills:int
    num_decodes:int
    original_max_key_length:int

def derive(common,route):
    if route is None:return None
    n=getattr(common,'num_reqs',None);nd=getattr(route,'num_decodes',None)
    total=getattr(common,'num_actual_tokens',None);maximum=getattr(common,'max_seq_len',None)
    if any(type(v) is not int for v in (n,nd,total,maximum)) or not 0<=nd<n or n>64 or total<=0 or maximum<=0:return None
    limits=getattr(common,'seq_lens_cpu_upper_bound',None);starts=getattr(common,'query_start_loc_cpu',None)
    for value in (limits,starts):
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.ndim!=1 or value.dtype not in (torch.int32,torch.int64):return None
    if limits.numel()<n or starts.numel()<n+1:return None
    start=starts[:n+1].tolist()
    if start[0]!=0 or start[-1]!=total:return None
    queries=[b-a for a,b in zip(start,start[1:])]
    if any(q!=1 for q in queries[:nd]) or any(q<=1 for q in queries[nd:]):return None
    keys=limits[nd:n].tolist()
    if any(k<q or k<=0 or k>maximum for k,q in zip(keys,queries[nd:])):return None
    bound=max(keys)
    if getattr(route,'max_prefill_query',None)!=max(queries[nd:]):return None
    return PrefillBound(bound,n-nd,nd,maximum)
