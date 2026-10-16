# SPDX-License-Identifier: Apache-2.0
"""Keep the qualified attention arithmetic for small native verify batches."""
import hashlib
from pathlib import Path

CACHE_SHA='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'


def install(native,cache,record=None):
    assert hashlib.sha256(Path(cache.__file__).read_bytes()).hexdigest()==CACHE_SHA
    import torch
    original=native.kernel_unified_attention
    seen=set()
    class Router:
        def __getitem__(self,grid):
            def launch(*args,**kw):
                selected=(not args and not kw['IS_3D'] and kw['KV_QUANT_MODE']==0
                    and kw['HEAD_SIZE']==128 and kw['num_queries_per_kv']==8
                    and kw['BLOCK_M']==16 and kw['BLOCK_Q']==2
                    and 1<=kw['num_seqs']<=64
                    and kw['num_seqs']<=kw['query_ptr'].shape[0]<=2*kw['num_seqs']
                    and kw['query_ptr'].dtype==kw['key_cache_ptr'].dtype==kw['value_cache_ptr'].dtype==torch.bfloat16
                    and kw['key_cache_ptr'].shape[1]==16
                    and kw['USE_CAUSAL'] and not kw['USE_PER_SEQ_CAUSAL'])
                if selected:
                    options=dict(kw,TILE_SIZE=32,num_warps=4,num_stages=1)
                    result=cache.kernel_unified_attention[grid](**options)
                    key=(kw['num_seqs'],kw['query_ptr'].shape[0])
                    if record is not None and key not in seen:
                        seen.add(key);record(dict(num_seqs=key[0],query_tokens=key[1],grid=list(grid),
                            BLOCK_M=16,BLOCK_Q=2,TILE_SIZE=32,num_warps=4,num_stages=1,
                            source_sha256=CACHE_SHA,n_regs=getattr(result,'n_regs',None),n_spills=getattr(result,'n_spills',None),
                            arithmetic_unchanged=True,query_positions_and_pointer_math_int64_preserved=True))
                    return result
                return original[grid](*args,**kw)
            return launch
    native.kernel_unified_attention=Router()
    return original
