# SPDX-License-Identifier: Apache-2.0
"""Separate exact scaling/BF16 projection and original SiLU after native INT32 GEMM."""
import torch,triton
import triton.language as tl
from flag_gems.fused.silu_and_mul import silu_and_mul_kernel as native_activation
_native_silu_scalar=native_activation._scalar_fn

@triton.jit
def _epilogue(Integer,AS,WS,O,Projected,M:tl.constexpr,I:tl.constexpr,BLOCK:tl.constexpr,DIAGNOSTICS:tl.constexpr):
    row=tl.program_id(0)
    cols=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    gate_int=tl.load(Integer+row*(2*I)+cols,cols<I,other=0)
    up_int=tl.load(Integer+row*(2*I)+I+cols,cols<I,other=0)
    scale_a=tl.load(AS+row)
    gate_scale=tl.load(WS+cols,cols<I,other=0.)
    up_scale=tl.load(WS+I+cols,cols<I,other=0.)
    gate=(gate_int.to(tl.float32)*scale_a*gate_scale).to(tl.bfloat16)
    up=(up_int.to(tl.float32)*scale_a*up_scale).to(tl.bfloat16)
    activated=_native_silu_scalar(gate,up)
    tl.store(O+row*I+cols,activated,cols<I)
    if DIAGNOSTICS:
        tl.store(Projected+row*(2*I)+cols,gate,cols<I)
        tl.store(Projected+row*(2*I)+I+cols,up,cols<I)

def epilogue(integer,ascale,wscale,out,projected,block,warps,diagnostics=False):
    m,n=integer.shape
    assert m==64 and n==12288 and integer.dtype==torch.int32 and integer.is_contiguous()
    assert ascale.dtype==wscale.dtype==torch.float32 and ascale.shape==(m,) and wscale.shape==(n,)
    assert out.dtype==torch.bfloat16 and out.shape==(m,n//2) and out.is_contiguous()
    assert block in (256,512,1024) and warps in (4,8)
    if diagnostics:assert projected.dtype==torch.bfloat16 and projected.shape==(m,n) and projected.is_contiguous()
    return _epilogue[(m,triton.cdiv(n//2,block))](integer,ascale,wscale,out,projected,m,n//2,block,diagnostics,num_warps=warps,num_stages=1)
