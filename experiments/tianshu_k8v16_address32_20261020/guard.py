# SPDX-License-Identifier: Apache-2.0
"""Metadata-only signed element-offset guard, with actual wide-kernel fallback."""
import hashlib
from pathlib import Path
import source

def inspect_bounds(kwargs):
    if not kwargs.get('IS_3D') or not 1<=kwargs.get('num_seqs',0)<=64 or kwargs.get('KV_QUANT_MODE')!=0:
        return False,[]
    if any(kwargs.get(name,False) for name in ('USE_TD','USE_TD_QO','USE_ALIBI_SLOPES','USE_QQ_BIAS','USE_SINKS','USE_MM_PREFIX')):
        return False,[]
    bounds=[]
    for name,value in kwargs.items():
        if isinstance(value,int) and not -2**31<=value<2**31:return False,bounds
        if not hasattr(value,'shape') or not callable(getattr(value,'stride',None)):continue
        shape=tuple(value.shape)
        if not shape:
            bound=0
        else:
            try:bound=source.offset_bound(shape,tuple(value.stride()))
            except (AssertionError,OverflowError,TypeError):return False,bounds
        bounds.append(dict(argument=name,signed_element_bound=bound))
        if not 0<=bound<2**31:return False,bounds
    return bool(bounds),bounds

class SelectingKernel:
    def __init__(self,control,candidate,control_sha,candidate_sha):
        self.control,self.candidate=control,candidate
        self.control_sha,self.candidate_sha=control_sha,candidate_sha
        self.last_selection=None
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            assert not args
            safe,bounds=inspect_bounds(kwargs)
            narrow=self.candidate is not self.control and safe
            active=self.candidate if narrow else self.control
            self.last_selection=dict(signed_offset_guard_passed=safe,address_bits=32 if narrow else 64,
                active_source_sha256=self.candidate_sha if narrow else self.control_sha,
                integer_bounds=bounds,wide_fallback_used=self.candidate is not self.control and not safe,
                original_arguments_forwarded=True,RNG_not_read_or_modified=True)
            return active[grid](**kwargs)
        return launch
