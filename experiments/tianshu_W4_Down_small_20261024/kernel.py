# SPDX-License-Identifier: Apache-2.0
import torch,triton
import triton.language as tl
@triton.jit
def _pad(X,Y,REAL:tl.constexpr,TOTAL:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    v=tl.load(X+i,i<REAL,other=0)
    tl.store(Y+i,v,i<TOTAL)
def run(x,w,s,z):
    m,k=x.shape;assert 1<=m<=64 and k==6144 and x.dtype==torch.bfloat16 and x.is_contiguous()
    if m in (1,2,32,64):return torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,w,128,s,z,size_n=2048)
    bucket=32 if m<32 else 64
    padded=torch.empty((bucket,k),device=x.device,dtype=x.dtype)
    _pad[(triton.cdiv(padded.numel(),1024),)](x,padded,x.numel(),padded.numel(),1024,num_warps=4,num_stages=1)
    return torch.ops.aten._weight_int4pack_mm_dotsaddz.default(padded,w,128,s,z,size_n=2048)[:m]
