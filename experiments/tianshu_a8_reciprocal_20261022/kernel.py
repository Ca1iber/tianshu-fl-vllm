# SPDX-License-Identifier: Apache-2.0
"""BF16 A8 reciprocal with explicit peak/2 correction; keep scale and integer codes."""
import torch
import triton
import triton.language as tl
from triton.language.extra.corex import libdevice
@triton.jit
def _quantize(A,Q,S,M:tl.constexpr,K:tl.constexpr,ASTRIDE:tl.constexpr,BLOCK:tl.constexpr,METHOD:tl.constexpr=2):
 row=tl.program_id(0);offsets=tl.arange(0,BLOCK);value=tl.load(A+row*ASTRIDE+offsets,offsets<K,other=0).to(tl.float32);peak=tl.max(tl.abs(value));scale=tl.div_rn(tl.maximum(peak,1.e-8),127.)
 if (peak>=1.e-8)&(peak<float('inf')):
  inverse=tl.div_rn(1.,scale);packed=libdevice.nearbyint(value*inverse)
  if METHOD==1:half=libdevice.nearbyint(tl.div_rn(peak*.5,scale))
  else:
   residual=tl.fma(-63.5,scale,peak*.5)
   half=tl.where(residual < -(scale*1.9073486328125e-6),63.,64.)
  packed=tl.where(tl.abs(value)==peak*.5,tl.where(value<0.,-half,half),packed)
 else:packed=libdevice.nearbyint(tl.div_rn(value,scale))
 packed=tl.minimum(tl.maximum(packed,-127.),127.).to(tl.int8);tl.store(Q+row*K+offsets,packed,offsets<K);tl.store(S+row,scale)
def quantize(a,q,scales,method=2):
 m,k=a.shape;assert a.dtype==torch.bfloat16 and q.dtype==torch.int8 and scales.dtype==torch.float32 and a.stride(1)==1 and q.shape==a.shape and q.is_contiguous() and scales.shape==(m,) and method in (1,2)
 return _quantize[(m,)](a,q,scales,m,k,a.stride(0),triton.next_power_of_2(k),method,num_warps=4,num_stages=1)
