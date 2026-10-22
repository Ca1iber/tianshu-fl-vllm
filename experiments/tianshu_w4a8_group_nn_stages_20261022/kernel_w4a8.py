# SPDX-License-Identifier: Apache-2.0
"""W4 storage decoded to INT8; native INT8 dot, FP32 group rescale. No INT4 dot claim."""
import torch
import triton
import triton.language as tl
@triton.jit
def _matmul(A,W,S,Z,AS,O,F,I,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,GROUP:tl.constexpr,AFFINE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,DIAGNOSTICS:tl.constexpr):
 pid=tl.program_id(0);tiles_n=tl.cdiv(N,BN);rows=(pid//tiles_n)*BM+tl.arange(0,BM);cols=(pid%tiles_n)*BN+tl.arange(0,BN);r=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
 for block in range(tl.cdiv(K,BK)):
  inner=block*BK+r;pairn=(pid%tiles_n)*(BN//2)+tl.arange(0,BN//2)
  a=tl.load(A+rows[:,None]*K+inner[None,:],(rows[:,None]<M)&(inner[None,:]<K),other=0)
  encoded=tl.load(W+inner[:,None]*tl.cdiv(N,2)+pairn[None,:],(inner[:,None]<K)&(pairn[None,:]<tl.cdiv(N,2)),other=0)
  qi=tl.interleave(encoded&15,encoded>>4).to(tl.int16);group=block*BK//GROUP
  if AFFINE:qi=qi-tl.load(Z+group*N+cols,cols<N,other=0).to(tl.int16)[None,:]
  else:qi=qi-8
  integer=tl.dot(a,qi.to(tl.int8),out_dtype=tl.int32)
  if DIAGNOSTICS:tl.store(I+block*M*N+rows[:,None]*N+cols[None,:],integer,(rows[:,None]<M)&(cols[None,:]<N))
  scale=tl.load(S+group*N+cols,cols<N,other=0.)
  acc=acc+integer.to(tl.float32)*scale[None,:]
 ascale=tl.load(AS+rows,rows<M,other=0.);value=acc*ascale[:,None];offsets=rows[:,None]*N+cols[None,:];valid=(rows[:,None]<M)&(cols[None,:]<N)
 if DIAGNOSTICS:tl.store(F+offsets,value,valid)
 tl.store(O+offsets,value,valid)
@triton.jit
def _decode(W,S,Z,O,N:tl.constexpr,K:tl.constexpr,GROUP:tl.constexpr,AFFINE:tl.constexpr,BLOCK:tl.constexpr):
 idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);n=idx//K;k=idx%K;mask=idx<N*K
 encoded=tl.load(W+k*tl.cdiv(N,2)+n//2,mask,other=0);qi=((encoded>>((n%2)*4))&15).to(tl.float32);group=k//GROUP
 if AFFINE:qi=qi-tl.load(Z+group*N+n,mask,other=0).to(tl.float32)
 else:qi=qi-8.
 scale=tl.load(S+group*N+n,mask,other=0.);tl.store(O+idx,(qi*scale).to(tl.bfloat16),mask)
def mm(a,w,scales,zero,ascale,out,float_out,integer,config,group=128,affine=False,diagnostics=False):
 m,k=a.shape;n=out.shape[1];assert a.dtype==torch.int8 and a.is_contiguous() and w.shape==(k,(n+1)//2) and w.is_contiguous();assert group>=config['bk'] and group%config['bk']==0
 assert scales.shape==((k+group-1)//group,n) and scales.is_contiguous() and ascale.shape==(m,) and ascale.dtype==scales.dtype==torch.float32
 assert out.shape==(m,n) and out.dtype==torch.bfloat16 and out.is_contiguous();assert not diagnostics or (float_out.shape==(m,n) and integer.shape==((k+config['bk']-1)//config['bk'],m,n) and integer.dtype==torch.int32)
 return _matmul[(triton.cdiv(m,config['bm'])*triton.cdiv(n,config['bn']),)](a,w,scales,zero,ascale,out,float_out,integer,m,n,k,group,affine,config['bm'],config['bn'],config['bk'],diagnostics,num_warps=config['warps'],num_stages=config['stages'])
def decode(w,scales,zero,out,bits=4,group=128,affine=False):
 n,k=out.shape;assert bits==4
 return _decode[(triton.cdiv(n*k,512),)](w,scales,zero,out,n,k,group,affine,512,num_warps=4,num_stages=1)
