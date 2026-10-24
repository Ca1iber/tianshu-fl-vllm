# SPDX-License-Identifier: Apache-2.0
"""K4 per-token/head RN32/RNE contract; BF16 V is copied without arithmetic."""
import torch
import triton
import triton.language as tl
from triton.language.extra.corex import libdevice

@triton.jit
def _write(Key,Value,K,V,S,Slots,
           KS:tl.constexpr,VS:tl.constexpr,KB:tl.constexpr,VB:tl.constexpr,
           KT:tl.constexpr,VT:tl.constexpr,KH:tl.constexpr,VH:tl.constexpr,
           SB:tl.constexpr,ST:tl.constexpr,SH:tl.constexpr):
    row=tl.program_id(0)
    head=tl.program_id(1)
    slot=tl.load(Slots+row).to(tl.int64)
    if slot<0:
        return
    d=tl.arange(0,128)
    key=tl.load(Key+row*KS+head*128+d).to(tl.float32)
    value=tl.load(Value+row*VS+head*128+d)
    maximum=tl.max(tl.abs(key),axis=0)
    scale=tl.where(maximum<7.888609052210118e-31,1.0,libdevice.div_rn(maximum,7.0))
    inverse=libdevice.div_rn(1.0,scale)
    q=tl.minimum(tl.maximum(libdevice.nearbyint(key*inverse),-7.0),7.0).to(tl.int16)
    codes=(q+8).to(tl.uint8)
    low,high=tl.split(codes.reshape(64,2))
    packed=low | (high << 4)
    block,token=slot//16,slot%16
    tl.store(K+block*KB+token*KT+head*KH+tl.arange(0,64),packed)
    tl.store(S+block*SB+token*ST+head*SH,scale)
    tl.store(V+block*VB+token*VT+head*VH+d,value)

def write(key,value,k,v,scales,slots):
    assert key.dtype==value.dtype==torch.bfloat16 and key.shape==value.shape
    assert key.shape[1:]==(2,128) and key.stride()[1:]==value.stride()[1:]==(128,1)
    assert k.dtype==torch.uint8 and k.shape[1:]==(16,2,64) and k.stride(-1)==1
    assert v.dtype==torch.bfloat16 and v.shape[1:]==(16,2,128) and v.stride(-1)==1
    assert scales.dtype==torch.float32 and scales.shape==(k.shape[0],16,2)
    assert slots.shape==(key.shape[0],) and slots.dtype==torch.int64
    return _write[(slots.numel(),2)](key,value,k,v,scales,slots,key.stride(0),value.stride(0),
        k.stride(0),v.stride(0),k.stride(1),v.stride(1),k.stride(2),v.stride(2),
        scales.stride(0),scales.stride(1),scales.stride(2),num_warps=1,num_stages=1,
        enable_fp_fusion=False)
