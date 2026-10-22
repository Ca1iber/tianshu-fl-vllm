# SPDX-License-Identifier: Apache-2.0
import torch,triton
import triton.language as tl
@triton.jit
def _dequant(Integer,AS,WS,O,M:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
 offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=offsets<M*N
 integer=tl.load(Integer+offsets,valid,other=0).to(tl.float32)
 asc=tl.load(AS+offsets//N,valid,other=0.);wsc=tl.load(WS+offsets%N,valid,other=0.)
 value=integer*asc*wsc
 tl.store(O+offsets,value,valid)
def dequant(integer,asc,wsc,out,block,warps):
 m,n=integer.shape
 assert (m,n)==(64,130560) and integer.dtype==torch.int32 and integer.is_contiguous()
 assert asc.shape==(m,) and wsc.shape==(n,) and asc.dtype==wsc.dtype==torch.float32
 assert out.shape==(m,n) and out.dtype==torch.bfloat16 and out.is_contiguous() and block in (256,512,1024,2048) and warps in (4,8)
 return _dequant[(triton.cdiv(m*n,block),)](integer,asc,wsc,out,m,n,block,num_warps=warps,num_stages=1)
