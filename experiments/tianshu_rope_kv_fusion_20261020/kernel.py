# SPDX-License-Identifier: Apache-2.0
"""Component only: native BF16-product Neox RoPE plus qualified K8/V16 store.

No model hook is installed. Return ordinary rotated Q/K and update the same
physical cache bytes the qualified post-RoPE writer would update.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _encode_bf16(data):
    bits=data.to(tl.uint16,bitcast=True).to(tl.uint32)
    magnitude=bits&32767
    rounded=(magnitude+7+((magnitude>>4)&1))>>4
    packed=tl.where(rounded>=936,tl.minimum(rounded-928,127),0)
    return (packed|((bits&32768)>>8)).to(tl.uint8)

@triton.jit
def fused(Positions,Q,K,V,CosSin,Slots,OQ,OK,KC,VC,
          QS:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,CS:tl.constexpr,
          KB:tl.constexpr,VB:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1)
    dim=tl.arange(0,64)
    pos=tl.load(Positions+row)
    c=tl.load(CosSin+pos*CS+dim).to(tl.float32)
    s=tl.load(CosSin+pos*CS+64+dim).to(tl.float32)
    if head<16:
        source=Q+row*QS+head*128
    else:
        source=K+row*KS+(head-16)*128
    x1=tl.load(source+dim).to(tl.float32)
    x2=tl.load(source+64+dim).to(tl.float32)
    # Current custom_ops=['none'] selects inherited forward_native. Its
    # four BF16 products round separately before the final subtraction/add.
    p1=(x1*c).to(tl.bfloat16).to(tl.float32)
    p2=(x2*s).to(tl.bfloat16).to(tl.float32)
    p3=(x2*c).to(tl.bfloat16).to(tl.float32)
    p4=(x1*s).to(tl.bfloat16).to(tl.float32)
    first=(p1-p2).to(tl.bfloat16)
    second=(p3+p4).to(tl.bfloat16)
    if head<16:
        tl.store(OQ+row*2048+head*128+dim,first)
        tl.store(OQ+row*2048+head*128+64+dim,second)
    else:
        h=head-16
        tl.store(OK+row*256+h*128+dim,first)
        tl.store(OK+row*256+h*128+64+dim,second)
        slot=tl.load(Slots+row).to(tl.int64)
        if slot>=0:
            block=slot//16;token=slot%16
            ko=block*KB+token*256+h*128
            vo=block*VB+token*256+h*128
            tl.store(KC+ko+dim,_encode_bf16(first))
            tl.store(KC+ko+64+dim,_encode_bf16(second))
            tl.store(VC+vo+dim,tl.load(V+row*VS+h*128+dim))
            tl.store(VC+vo+64+dim,tl.load(V+row*VS+h*128+64+dim))


def launch(positions,q,k,v,cache,slots,oq,ok,kc,vc,warps=1):
    assert q.dtype==k.dtype==v.dtype==cache.dtype==oq.dtype==ok.dtype==vc.dtype==torch.bfloat16
    assert kc.dtype==torch.uint8 and positions.dtype==slots.dtype==torch.int64
    n=q.shape[0]
    assert 1<=n<=64 and q.shape==(n,2048) and k.shape==v.shape==(n,256)
    assert oq.shape==q.shape and ok.shape==k.shape and oq.is_contiguous() and ok.is_contiguous()
    assert positions.shape==slots.shape==(n,) and positions.is_contiguous() and slots.is_contiguous()
    assert cache.ndim==2 and cache.shape[1]==128 and cache.stride(1)==1
    assert all(x.stride(1)==1 for x in (q,k,v))
    assert kc.shape==vc.shape and kc.shape[1:]==(16,2,128)
    assert kc.stride()[1:]==vc.stride()[1:]==(256,128,1)
    assert kc.stride(0)==2*vc.stride(0)==12288
    assert warps in (1,2,4)
    return fused[(n,18)](positions,q,k,v,cache,slots,oq,ok,kc,vc,
        q.stride(0),k.stride(0),v.stride(0),cache.stride(0),kc.stride(0),vc.stride(0),
        num_warps=warps,num_stages=1,enable_fp_fusion=False)
