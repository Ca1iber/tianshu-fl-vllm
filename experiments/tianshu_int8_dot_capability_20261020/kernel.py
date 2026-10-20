# SPDX-License-Identifier: Apache-2.0
"""Capability only: rowwise Q quantization and exact INT8 QK matrix product."""
import triton
import triton.language as tl
from triton.language.extra.corex import libdevice

@triton.jit
def qk(Q,K,Scale,Q8,QS,Integer,Scaled,N:tl.constexpr):
    rows=tl.arange(0,16)
    inner=tl.arange(0,128)
    columns=tl.arange(0,N)
    q=tl.load(Q+rows[:,None]*128+inner[None,:]).to(tl.float32)
    maximum=tl.maximum(tl.max(tl.abs(q),axis=1),1.e-8)
    scale=tl.div_rn(maximum,127.)
    rounded=libdevice.nearbyint(tl.div_rn(q,scale[:,None]))
    packed=tl.minimum(tl.maximum(rounded,-127.),127.).to(tl.int8)
    k=tl.load(K+inner[:,None]*N+columns[None,:])
    result=tl.dot(packed,k,out_dtype=tl.int32)
    key_scale=tl.load(Scale+columns)
    floating=result.to(tl.float32)*scale[:,None]*key_scale[None,:]
    tl.store(Q8+rows[:,None]*128+inner[None,:],packed)
    tl.store(QS+rows,scale)
    tl.store(Integer+rows[:,None]*N+columns[None,:],result)
    tl.store(Scaled+rows[:,None]*N+columns[None,:],floating)
