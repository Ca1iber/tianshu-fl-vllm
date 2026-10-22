# SPDX-License-Identifier: Apache-2.0
import torch,triton
import triton.language as tl
from flag_gems.fused.silu_and_mul import silu_and_mul_kernel as native_activation
_native_silu_scalar=native_activation._scalar_fn
@triton.jit
def _epilogue(Integer,AS,WS,O,Projected,M:tl.constexpr,I:tl.constexpr,ROWS:tl.constexpr,DIAGNOSTICS:tl.constexpr):
 rows=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
 columns=tl.program_id(1)*1024+tl.arange(0,1024)
 valid=(rows[:,None]<M)&(columns[None,:]<I)
 gate_int=tl.load(Integer+rows[:,None]*(2*I)+columns[None,:],valid,other=0)
 up_int=tl.load(Integer+rows[:,None]*(2*I)+I+columns[None,:],valid,other=0)
 asc=tl.load(AS+rows,rows<M,other=0.)
 gs=tl.load(WS+columns,columns<I,other=0.)
 us=tl.load(WS+I+columns,columns<I,other=0.)
 gate=(gate_int.to(tl.float32)*asc[:,None]*gs[None,:]).to(tl.bfloat16)
 up=(up_int.to(tl.float32)*asc[:,None]*us[None,:]).to(tl.bfloat16)
 out=_native_silu_scalar(gate,up)
 tl.store(O+rows[:,None]*I+columns[None,:],out,valid)
 if DIAGNOSTICS:
  tl.store(Projected+rows[:,None]*(2*I)+columns[None,:],gate,valid)
  tl.store(Projected+rows[:,None]*(2*I)+I+columns[None,:],up,valid)
def epilogue(integer,asc,wsc,out,projected,rows,diagnostics=False):
 m,n=integer.shape
 assert m in (128,512,4096) and n==12288 and integer.dtype==torch.int32 and integer.is_contiguous()
 assert asc.dtype==wsc.dtype==torch.float32 and asc.shape==(m,) and wsc.shape==(n,)
 assert out.dtype==torch.bfloat16 and out.shape==(m,n//2) and out.is_contiguous() and rows in (1,4)
 if diagnostics:assert projected.dtype==torch.bfloat16 and projected.shape==(m,n) and projected.is_contiguous()
 return _epilogue[(triton.cdiv(m,rows),6)](integer,asc,wsc,out,projected,m,n//2,rows,diagnostics,num_warps=8,num_stages=1)
