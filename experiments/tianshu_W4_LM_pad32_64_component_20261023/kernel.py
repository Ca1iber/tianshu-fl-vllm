# SPDX-License-Identifier: Apache-2.0
import torch,triton
import triton.language as tl
@triton.jit
def _pad(X,Y,REAL:tl.constexpr,TOTAL:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<REAL,other=0)
    tl.store(Y+i,x,i<TOTAL)
def pad(x,out):
    assert x.is_contiguous() and out.is_contiguous() and x.shape[1]==out.shape[1]==2048 and out.shape[0]>=x.shape[0]
    return _pad[(triton.cdiv(out.numel(),1024),)](x,out,x.numel(),out.numel(),1024,num_warps=4,num_stages=1)
def run(x,w,s,z):
    m=x.shape[0];assert 1<=m<=64 and x.dtype==torch.bfloat16 and x.is_contiguous()
    b=32 if m<=32 else 64
    if m==b:return torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,w,128,s,z,size_n=130560)
    padded=torch.empty((b,2048),device=x.device,dtype=x.dtype);pad(x,padded)
    return torch.ops.aten._weight_int4pack_mm_dotsaddz.default(padded,w,128,s,z,size_n=130560)[:m]
