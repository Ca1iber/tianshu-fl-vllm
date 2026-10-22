# SPDX-License-Identifier: Apache-2.0
"""Fixed 128-channel BF16 correction for a zero-masked native A8W8 Down."""
import torch,triton
import triton.language as tl
from triton.language.extra.corex import libdevice

@triton.jit
def _quantize_bulk(X,Mask,Q,S,M:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0);inner=tl.arange(0,BLOCK)
 value=tl.load(X+row*K+inner,inner<K,other=0).to(tl.float32)
 is_side=tl.load(Mask+inner,inner<K,other=0)
 value=tl.where(is_side!=0,0.,value)
 scale=tl.div_rn(tl.maximum(tl.max(tl.abs(value)),1.e-8),127.)
 packed=libdevice.nearbyint(tl.div_rn(value,scale));packed=tl.minimum(tl.maximum(packed,-127.),127.).to(tl.int8)
 tl.store(Q+row*K+inner,packed,inner<K);tl.store(S+row,scale)

@triton.jit
def _correct(Integer,AS,WS,X,Indices,SideWeight,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,SIDE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
 rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.program_id(1)*BN+tl.arange(0,BN);inner=tl.arange(0,SIDE)
 selected=tl.load(Indices+inner)
 left=tl.load(X+rows[:,None]*K+selected[None,:],rows[:,None]<M,other=0)
 right=tl.load(SideWeight+inner[:,None]*N+cols[None,:],cols[None,:]<N,other=0)
 correction=tl.dot(left,right,out_dtype=tl.float32)
 offsets=rows[:,None]*N+cols[None,:];valid=(rows[:,None]<M)&(cols[None,:]<N)
 integer=tl.load(Integer+offsets,valid,other=0).to(tl.float32)
 asc=tl.load(AS+rows,rows<M,other=0.);wsc=tl.load(WS+cols,cols<N,other=0.)
 value=(integer*asc[:,None])*wsc[None,:]+correction
 tl.store(O+offsets,value,valid)

@triton.jit
def _reference_side(X,Indices,SideWeight,Side,AbsSum,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,SIDE:tl.constexpr):
 rows=tl.program_id(0)*32+tl.arange(0,32);cols=tl.program_id(1)*64+tl.arange(0,64);inner=tl.arange(0,32)
 acc=tl.zeros((32,64),tl.float32);absolute=tl.zeros((32,64),tl.float32)
 for block in range(SIDE//32):
  selected=tl.load(Indices+block*32+inner)
  left=tl.load(X+rows[:,None]*K+selected[None,:],rows[:,None]<M,other=0)
  right=tl.load(SideWeight+(block*32+inner[:,None])*N+cols[None,:],cols[None,:]<N,other=0)
  acc=tl.dot(left,right,acc,out_dtype=tl.float32)
  absolute=tl.dot(tl.abs(left),tl.abs(right),absolute,out_dtype=tl.float32)
 offsets=rows[:,None]*N+cols[None,:];valid=(rows[:,None]<M)&(cols[None,:]<N)
 tl.store(Side+offsets,acc,valid);tl.store(AbsSum+offsets,absolute,valid)

@triton.jit
def _reference_combine(Integer,AS,WS,Side,AbsSum,Reference,Bound,M:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
 offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=offsets<M*N
 integer=tl.load(Integer+offsets,valid,other=0).to(tl.float32)
 asc=tl.load(AS+offsets//N,valid,other=0.);wsc=tl.load(WS+offsets%N,valid,other=0.)
 bulk=(integer*asc)*wsc;side=tl.load(Side+offsets,valid,other=0.);absolute=tl.load(AbsSum+offsets,valid,other=0.)
 value=bulk+side
 # Half-BF16 relative rounding bound plus gamma_128 accumulation in each side path.
 error=0.00390625*tl.abs(value)+0.0000305181*absolute+0.0000005*(tl.abs(bulk)+tl.abs(side))+1.e-7
 tl.store(Reference+offsets,value,valid);tl.store(Bound+offsets,error,valid)

def quantize(x,mask,q,scale):
 m,k=x.shape;assert k==6144 and x.dtype==torch.bfloat16 and x.is_contiguous() and mask.shape==(k,) and mask.dtype==torch.uint8 and q.shape==x.shape and q.dtype==torch.int8 and scale.shape==(m,) and scale.dtype==torch.float32
 return _quantize_bulk[(m,)](x,mask,q,scale,m,k,8192,num_warps=4,num_stages=1,enable_fp_fusion=False)

def correct(integer,asc,wsc,x,indices,sideweight,out):
 m,n=integer.shape;assert n==2048 and x.shape==(m,6144) and indices.shape==(128,) and sideweight.shape==(128,n) and out.shape==(m,n)
 assert integer.dtype==torch.int32 and asc.dtype==wsc.dtype==torch.float32 and x.dtype==sideweight.dtype==out.dtype==torch.bfloat16 and indices.dtype==torch.int32
 assert all(t.is_contiguous() for t in (integer,asc,wsc,x,indices,sideweight,out))
 return _correct[(triton.cdiv(m,16),triton.cdiv(n,64))](integer,asc,wsc,x,indices,sideweight,out,m,n,6144,128,16,64,num_warps=4,num_stages=1,enable_fp_fusion=False)

def reference(integer,asc,wsc,x,indices,sideweight,side,absolute,output,bound):
 m,n=integer.shape
 _reference_side[(triton.cdiv(m,32),triton.cdiv(n,64))](x,indices,sideweight,side,absolute,m,n,6144,128,num_warps=4,num_stages=1,enable_fp_fusion=False)
 return _reference_combine[(triton.cdiv(m*n,1024),)](integer,asc,wsc,side,absolute,output,bound,m,n,1024,num_warps=4,num_stages=1,enable_fp_fusion=False)
