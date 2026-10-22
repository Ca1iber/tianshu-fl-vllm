# SPDX-License-Identifier: Apache-2.0
import triton,triton.language as tl
from triton.language.extra.corex import libdevice
@triton.jit
def _diagnostic(A,W,AS,WS,O,Integers,Acc,Scaled,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,GROUP:tl.constexpr,GROUPS:tl.constexpr,RN:tl.constexpr):
 pid=tl.program_id(0);rows=(pid//16)*32+tl.arange(0,32);cols=(pid%16)*128+tl.arange(0,128);inner=tl.arange(0,128);offset=rows[:,None]*N+cols[None,:];valid=(rows[:,None]<M)&(cols[None,:]<N)
 total=tl.zeros((32,128),tl.float32)
 for group in range(GROUPS):
  integer=tl.zeros((32,128),tl.int32)
  for part in tl.static_range(GROUP//128):
   absolute=group*GROUP+part*128+inner
   left=tl.load(A+rows[:,None]*K+absolute[None,:],rows[:,None]<M,other=0)
   right=tl.load(W+cols[None,:]*K+absolute[:,None],cols[None,:]<N,other=0)
   integer=tl.dot(left,right,integer,out_dtype=tl.int32)
  tl.store(Integers+group*M*N+offset,integer,valid)
  asc=tl.load(AS+rows*GROUPS+group,rows<M,other=0.)
  if RN:total=libdevice.add_rn(total,libdevice.mul_rn(integer.to(tl.float32),asc[:,None]))
  else:total=total+integer.to(tl.float32)*asc[:,None]
 wsc=tl.load(WS+cols,cols<N,other=0.)
 if RN:scaled=libdevice.mul_rn(total,wsc[None,:])
 else:scaled=total*wsc[None,:]
 tl.store(Acc+offset,total,valid);tl.store(Scaled+offset,scaled,valid);tl.store(O+offset,scaled,valid)
def run(q,w,asc,ws,out,integers,acc,scaled,group,warps,rn):
 return _diagnostic[(32,)](q,w,asc,ws,out,integers,acc,scaled,64,2048,6144,group,6144//group,rn,num_warps=warps,num_stages=1,enable_fp_fusion=False)
