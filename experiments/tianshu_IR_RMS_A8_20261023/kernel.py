# SPDX-License-Identifier: Apache-2.0
"""Functional native-IR RMSNorm + exact dynamic symmetric A8 packing."""
import torch
import triton
import triton.language as tl
from triton.language.extra.corex import libdevice

@triton.jit
def _norm_a8(X,R,W,Y,RY,Q,S,K:tl.constexpr,EPS:tl.constexpr,HAS_R:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0);col=tl.arange(0,BLOCK);mask=col<K
 value=tl.load(X+row*K+col,mask,other=0).to(tl.float32)
 if HAS_R:
  value=value+tl.load(R+row*K+col,mask,other=0).to(tl.float32)
  tl.store(RY+row*K+col,value.to(tl.bfloat16),mask)
 variance=tl.sum(value*value,0)/K
 inv=tl.rsqrt(variance+EPS)
 normalized=(value*inv).to(tl.bfloat16).to(tl.float32)
 weight=tl.load(W+col,mask,other=0).to(tl.float32)
 y=(normalized*weight).to(tl.bfloat16)
 tl.store(Y+row*K+col,y,mask)
 quant_input=y.to(tl.float32)
 scale=tl.div_rn(tl.maximum(tl.max(tl.abs(quant_input),0),1.e-8),127.)
 quant=libdevice.nearbyint(tl.div_rn(quant_input,scale))
 quant=tl.minimum(tl.maximum(quant,-127.),127.).to(tl.int8)
 tl.store(Q+row*K+col,quant,mask);tl.store(S+row,scale)

def run(x,residual,weight,y,residual_out,q,scale):
 m,k=x.shape;has=residual is not None
 assert k==2048 and x.dtype==weight.dtype==y.dtype==torch.bfloat16
 assert x.is_contiguous() and weight.shape==(k,) and weight.is_contiguous()
 assert y.shape==q.shape==x.shape and y.is_contiguous() and q.is_contiguous() and q.dtype==torch.int8
 assert scale.shape==(m,) and scale.dtype==torch.float32 and scale.is_contiguous()
 if has:assert residual.shape==residual_out.shape==x.shape and residual.dtype==residual_out.dtype==torch.bfloat16 and residual.is_contiguous() and residual_out.is_contiguous()
 return _norm_a8[(m,)](x,residual if has else x,weight,y,residual_out if has else y,q,scale,k,1.e-6,has,triton.next_power_of_2(k),num_warps=4,num_stages=1)
