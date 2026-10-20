# SPDX-License-Identifier: Apache-2.0
"""Dynamic K int8 plus FP32 token/head scale; V remains bit-preserving BF16."""
import torch
import triton
import triton.language as tl
from triton.language.extra.corex import libdevice

@triton.jit
def kernel(K,V,CacheK,CacheV,Slots,KS0:tl.int64,KS1:tl.int64,KS2:tl.int64,
           VS0:tl.int64,VS1:tl.int64,VS2:tl.int64,PageK:tl.int64,PageV:tl.int64):
    row=tl.program_id(0);head=tl.program_id(1)
    slot=tl.load(Slots+row).to(tl.int64)
    if slot<0:return
    offsets=tl.arange(0,128)
    key=tl.load(K+row*KS0+head*KS1+offsets*KS2).to(tl.float32)
    value=tl.load(V+row*VS0+head*VS1+offsets*VS2)
    maximum=tl.maximum(tl.max(tl.abs(key),0),1.e-8)
    scale=tl.div_rn(maximum,127.)
    rounded=libdevice.nearbyint(tl.div_rn(key,scale))
    code=tl.minimum(tl.maximum(rounded,-127.),127.).to(tl.int8).to(tl.uint8,bitcast=True)
    page=slot//16;position=slot%16
    tl.store(CacheK+page*PageK+position*256+head*128+offsets,code)
    tl.store(CacheV+page*PageV+position*256+head*128+offsets,value)
    scales=(CacheK+page*PageK+12288).to(tl.pointer_type(tl.float32))
    tl.store(scales+position*2+head,scale)

@triton.jit
def decode_kernel(CacheK,Slots,Out,PageK:tl.int64,OS0:tl.int64,OS1:tl.int64,OS2:tl.int64):
    row=tl.program_id(0);head=tl.program_id(1)
    slot=tl.load(Slots+row).to(tl.int64)
    if slot<0:return
    offsets=tl.arange(0,128);page=slot//16;position=slot%16
    raw=tl.load(CacheK+page*PageK+position*256+head*128+offsets)
    scales=(CacheK+page*PageK+12288).to(tl.pointer_type(tl.float32))
    scale=tl.load(scales+position*2+head)
    value=(raw.to(tl.int8,bitcast=True).to(tl.float32)*scale).to(tl.bfloat16)
    tl.store(Out+row*OS0+head*OS1+offsets*OS2,value)

def write(key,value,k,v,slots):
    assert key.dtype==value.dtype==v.dtype==torch.bfloat16 and k.dtype==torch.uint8 and slots.dtype==torch.int64
    assert key.shape==value.shape and key.ndim==3 and tuple(key.shape[1:])==(2,128)
    assert key.shape[0]==slots.numel()
    assert k.ndim==v.ndim==4 and k.shape==v.shape and tuple(k.shape[1:])==(16,2,128)
    assert k.stride()[1:]==v.stride()[1:]==(256,128,1)
    assert k.stride(0)%12416==0 and v.stride(0)*2==k.stride(0)
    assert v.data_ptr()==k.data_ptr()+4096
    assert k.storage_offset()*k.element_size()+(k.shape[0]-1)*k.stride(0)+12416<=k.untyped_storage().nbytes()
    return kernel[(slots.numel(),2)](key,value,k,v,slots,*key.stride(),*value.stride(),k.stride(0),v.stride(0),num_warps=1,num_stages=1)

def decode(k,slots,out):
    assert out.dtype==torch.bfloat16 and out.shape==(slots.numel(),2,128)
    return decode_kernel[(slots.numel(),2)](k,slots,out,k.stride(0),*out.stride(),num_warps=1,num_stages=1)
