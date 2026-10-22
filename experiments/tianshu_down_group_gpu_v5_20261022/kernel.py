# SPDX-License-Identifier: Apache-2.0
"""Dynamic A8 per K group, fixed W8 per output, FP32 group accumulation."""
import torch,triton
import triton.language as tl
from triton.language.extra.corex import libdevice
@triton.jit
def _quantize_m64_v5(X,Q,AS,M:tl.constexpr,K:tl.constexpr,GROUP:tl.constexpr,GROUPS:tl.constexpr,PAD_GROUPS:tl.constexpr):
 row=tl.program_id(0);groups=tl.arange(0,PAD_GROUPS);inner=tl.arange(0,GROUP)
 absolute=groups[:,None]*GROUP+inner[None,:];valid=groups[:,None]<GROUPS
 value=tl.load(X+row*K+absolute,valid,other=0).to(tl.float32)
 scale=tl.div_rn(tl.maximum(tl.max(tl.abs(value),1),1.e-8),127.)
 packed=libdevice.nearbyint(tl.div_rn(value,scale[:,None]));packed=tl.minimum(tl.maximum(packed,-127.),127.).to(tl.int8)
 tl.store(Q+row*K+absolute,packed,valid);tl.store(AS+row*GROUPS+groups,scale,groups<GROUPS)
@triton.jit
def _gemm_m64_v5(A,W,AS,WS,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,GROUP:tl.constexpr,GROUPS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,NN:tl.constexpr):
 tiles_n=tl.cdiv(N,BN);pid=tl.program_id(0);rows=(pid//tiles_n)*BM+tl.arange(0,BM);cols=(pid%tiles_n)*BN+tl.arange(0,BN);inner=tl.arange(0,BK)
 total=tl.zeros((BM,BN),tl.float32)
 for group in range(GROUPS):
  integer=tl.zeros((BM,BN),tl.int32)
  for part in tl.static_range(GROUP//BK):
   absolute=group*GROUP+part*BK+inner
   left=tl.load(A+rows[:,None]*K+absolute[None,:],rows[:,None]<M,other=0)
   if NN:right=tl.load(W+absolute[:,None]*N+cols[None,:],cols[None,:]<N,other=0)
   else:right=tl.load(W+cols[None,:]*K+absolute[:,None],cols[None,:]<N,other=0)
   integer=tl.dot(left,right,integer,out_dtype=tl.int32)
  asc=tl.load(AS+rows*GROUPS+group,rows<M,other=0.)
  total=libdevice.fma_rn(integer.to(tl.float32),asc[:,None],total)
 wsc=tl.load(WS+cols,cols<N,other=0.)
 tl.store(O+rows[:,None]*N+cols[None,:],total*wsc[None,:],(rows[:,None]<M)&(cols[None,:]<N))
def quantize(x,q,asc,group):
 m,k=x.shape;assert k==6144 and group in (128,512) and x.dtype==torch.bfloat16 and q.dtype==torch.int8 and asc.dtype==torch.float32 and asc.shape==(m,k//group) and x.is_contiguous() and q.is_contiguous() and asc.is_contiguous()
 return _quantize_m64_v5[(m,)](x,q,asc,m,k,group,k//group,triton.next_power_of_2(k//group),num_warps=4,num_stages=1,enable_fp_fusion=False)
def mm(q,w,asc,wsc,out,group,warps,bm,bn,layout,stages):
 m,k=q.shape;n=2048;assert m==64 and k==6144 and group in (128,512) and warps in (8,16) and (bm,bn) in ((32,128),(64,64)) and layout in ('NT','NN')
 assert w.shape==((k,n) if layout=='NN' else (n,k)) and q.dtype==w.dtype==torch.int8 and wsc.dtype==asc.dtype==torch.float32 and out.dtype==torch.bfloat16 and asc.shape==(m,k//group) and wsc.shape==(n,) and out.shape==(m,n) and all(t.is_contiguous() for t in (q,w,asc,wsc,out))
 return _gemm_m64_v5[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](q,w,asc,wsc,out,m,n,k,group,k//group,bm,bn,128,layout=='NN',num_warps=warps,num_stages=stages,enable_fp_fusion=False)

@triton.jit
def _reference_fma_m64_v5(Integers,AS,WS,O,M:tl.constexpr,N:tl.constexpr,GROUPS:tl.constexpr,BLOCK:tl.constexpr):
 offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=offsets<M*N;rows=offsets//N
 total=tl.full((BLOCK,),0.,tl.float32)
 for group in range(GROUPS):
  integer=tl.load(Integers+group*M*N+offsets,valid,other=0).to(tl.float32)
  asc=tl.load(AS+rows*GROUPS+group,valid,other=0.)
  total=libdevice.fma_rn(integer,asc,total)
 wsc=tl.load(WS+offsets%N,valid,other=0.)
 tl.store(O+offsets,libdevice.mul_rn(total,wsc),valid)
def reference_fma(integer_groups,asc,wsc,out):
 groups,m,n=integer_groups.shape
 assert integer_groups.dtype==torch.int32 and integer_groups.is_contiguous() and asc.shape==(m,groups) and wsc.shape==(n,) and out.shape==(m,n)
 return _reference_fma_m64_v5[(triton.cdiv(m*n,1024),)](integer_groups,asc,wsc,out,m,n,groups,1024,num_warps=4,num_stages=1,enable_fp_fusion=False)
