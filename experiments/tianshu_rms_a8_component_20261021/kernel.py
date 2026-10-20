# SPDX-License-Identifier: Apache-2.0
"""Component only: current compiled FP32 weighted RMS -> BF16 -> row A8.

Materialize normalized BF16 too, so a future explicit FX route can preserve a
BF16 fallback. This has no model integration or implicit global tensor cache.
"""
import triton
import triton.language as tl
from triton.language.extra.corex import libdevice

@triton.jit
def rms_a8(X,R,W,Norm,Q,Scale,M:tl.constexpr,H:tl.constexpr,ROWS:tl.constexpr):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    col=tl.arange(0,H)
    valid=row[:,None]<M
    x=tl.load(X+row[:,None]*H+col[None,:],valid,other=0).to(tl.float32)
    r=tl.load(R+row[:,None]*H+col[None,:],valid,other=0).to(tl.float32)
    w=tl.load(W+col).to(tl.float32)
    combined=x+r
    variance=tl.sum(combined*combined,axis=1)/H
    inv=libdevice.rsqrt(variance+1.e-6)
    # Deliberately matches actual generated model code, where Inductor elides
    # the intermediate BF16 cast before weight multiplication. The native
    # Python IR contains that cast; it must not be mistaken for current code.
    norm=(combined*inv[:,None]*w[None,:]).to(tl.bfloat16)
    value=norm.to(tl.float32)
    scale=tl.div_rn(tl.maximum(tl.max(tl.abs(value),axis=1),1.e-8),127.)
    quant=libdevice.nearbyint(tl.div_rn(value,scale[:,None]))
    quant=tl.minimum(tl.maximum(quant,-127.),127.).to(tl.int8)
    tl.store(Norm+row[:,None]*H+col[None,:],norm,valid)
    tl.store(Q+row[:,None]*H+col[None,:],quant,valid)
    tl.store(Scale+row,scale,row<M)
