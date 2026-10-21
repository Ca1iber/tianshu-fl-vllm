# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch only: preserve the original JIT and every non-target call."""
from contextlib import contextmanager

def eligible(logits,kw):
    return (logits.device.type=='cuda' and str(logits.dtype)=='torch.float32'
        and len(logits.shape)==2 and logits.shape[1]==130560 and 8<=logits.shape[0]<=64
        and logits.stride(1)==1 and logits.stride(0)==130560
        and kw.get('TOPK_ENABLED') is False and kw.get('TOPP_ENABLED') is True
        and kw.get('BLOCK_SIZE')==8192 and kw.get('BLOCK_SIZE_TRUNC')==4096
        and kw.get('VOCAB_SIZE')==130560 and kw.get('BATCH_SIZE')==logits.shape[0])

class KernelRouter:
    def __init__(self,original,selection,observe):self.original=original;self.selection=selection;self.observe=observe
    def __getattr__(self,name):return getattr(self.original,name)
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            matches=eligible(args[0],kwargs);chosen=self.selection() if matches else None
            effective=dict(kwargs,num_warps=chosen) if chosen is not None else kwargs
            answer=self.original[grid](*args,**effective)
            if matches:self.observe(grid,args,effective,answer)
            return answer
        return launch

def profile_with_warmup(original,warm,finished):
    @contextmanager
    def profiling(*args,**kwargs):
        with original(*args,**kwargs) as result:
            warm(result)
            yield result
        finished(result)
    return profiling
