# SPDX-License-Identifier: Apache-2.0
"""Fuse original FP32 in-place division with a first-stage exact argmax."""
import torch
import triton
import triton.language as tl

@triton.jit
def partial(P,Q,V,I,N:tl.constexpr,PS:tl.constexpr,QS:tl.constexpr,TILES:tl.constexpr,TILE:tl.constexpr):
    row,block=tl.program_id(0),tl.program_id(1)
    col=block*TILE+tl.arange(0,TILE);valid=col<N
    p=tl.load(P+row*PS+col,valid,other=0.)
    q=tl.load(Q+row*QS+col,valid,other=1.)
    value=p/q
    tl.store(P+row*PS+col,value,valid)
    nan=valid&(value!=value)
    first_nan=tl.min(tl.where(nan,col,2147483647),0)
    maximum=tl.max(tl.where(valid&~nan,value,-float('inf')),0)
    first_max=tl.min(tl.where(valid&(value==maximum),col,2147483647),0)
    has_nan=first_nan!=2147483647
    tl.store(V+row*TILES+block,tl.where(has_nan,float('nan'),maximum))
    tl.store(I+row*TILES+block,tl.where(has_nan,first_nan,first_max))

@triton.jit
def finish(V,I,O,TILES:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK);valid=col<TILES
    value=tl.load(V+row*TILES+col,valid,other=-float('inf'))
    index=tl.load(I+row*TILES+col,valid,other=2147483647)
    nan=valid&(value!=value)
    first_nan=tl.min(tl.where(nan,index,2147483647),0)
    maximum=tl.max(tl.where(nan,-float('inf'),value),0)
    first_max=tl.min(tl.where(valid&(value==maximum),index,2147483647),0)
    tl.store(O+row,tl.where(first_nan!=2147483647,first_nan,first_max).to(tl.int64))

def launch(probs,noise):
    assert probs.is_cuda and probs.ndim==2 and probs.dtype==noise.dtype==torch.float32
    assert probs.shape==noise.shape and probs.stride(1)==noise.stride(1)==1 and probs.device==noise.device
    b,n=probs.shape;tiles=triton.cdiv(n,4096)
    v=torch.empty((b,tiles),device=probs.device,dtype=torch.float32)
    i=torch.empty((b,tiles),device=probs.device,dtype=torch.int32)
    out=torch.empty((b,),device=probs.device,dtype=torch.int64)
    partial[(b,tiles)](probs,noise,v,i,n,probs.stride(0),noise.stride(0),tiles,4096,num_warps=4,num_stages=1)
    finish[(b,)](v,i,out,tiles,triton.next_power_of_2(tiles),num_warps=4,num_stages=1)
    return out
