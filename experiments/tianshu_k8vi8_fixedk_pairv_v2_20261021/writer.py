# SPDX-License-Identifier: Apache-2.0
"""Fixed software8 K and per-token/head INT8 V; real scales share each page."""
import hashlib,importlib.util
from pathlib import Path
import torch,triton
import triton.language as tl
from triton.language.extra.corex import libdevice
PARENT=Path(__file__).resolve().parent.parent/'tianshu_soft8_mixed_planes_20261019/writer.py'
spec=importlib.util.spec_from_file_location('_kvi8_original_writer',PARENT)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
encode_all=old.encode_all

@triton.jit
def _write_heads(Key,Value,K,V,KS,VS,Slots,INPUT_KS:tl.constexpr,INPUT_VS:tl.constexpr,
                 KB:tl.constexpr,VB:tl.constexpr,KSB:tl.constexpr,VSB:tl.constexpr):
    row=tl.program_id(0)
    slot=tl.load(Slots+row).to(tl.int64)
    if slot<0:return
    head=tl.arange(0,2)
    dim=tl.arange(0,128)
    offsets=head[:,None]*128+dim[None,:]
    key=tl.load(Key+row*INPUT_KS+offsets)
    value=tl.load(Value+row*INPUT_VS+offsets).to(tl.float32)
    scale=tl.div_rn(tl.maximum(tl.max(tl.abs(value),axis=1),1.e-8),127.)
    quant=libdevice.nearbyint(tl.div_rn(value,scale[:,None]))
    quant=tl.minimum(tl.maximum(quant,-127.),127.).to(tl.int8)
    block,token=slot//16,slot%16
    tl.store(K+block*KB+token*256+offsets,old._soft8_encode_bf16(key))
    tl.store(V+block*VB+token*256+offsets,quant)
    tl.store(KS+block*KSB+token*2+head,1.)
    tl.store(VS+block*VSB+token*2+head,scale)

def write(key,value,k,v,slots,ks=None,vs=None):
    if v.dtype!=torch.int8:
        assert ks is None and vs is None
        return old.write(key,value,k,v,slots)
    assert key.dtype==value.dtype==torch.bfloat16 and key.shape==value.shape and key.shape[1:]==(2,128)
    assert key.stride(1)==value.stride(1)==128 and key.stride(2)==value.stride(2)==1
    assert k.dtype==torch.uint8 and k.shape==v.shape and k.shape[1:]==(16,2,128)
    assert ks.dtype==vs.dtype==torch.float32 and ks.shape==vs.shape==k.shape[:3]
    assert k.stride()[1:]==v.stride()[1:]==(256,128,1)
    assert ks.stride()[1:]==vs.stride()[1:]==(2,1) and slots.shape==(key.shape[0],)
    return _write_heads[(slots.numel(),)](key,value,k,v,ks,vs,slots,key.stride(0),value.stride(0),
        k.stride(0),v.stride(0),ks.stride(0),vs.stride(0),num_warps=1,num_stages=1)
