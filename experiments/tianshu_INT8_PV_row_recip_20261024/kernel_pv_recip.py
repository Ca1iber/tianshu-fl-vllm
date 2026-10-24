# SPDX-License-Identifier: Apache-2.0
"""Per-row exact reciprocal followed by FP32 multiply; explicit changed rounding; INT32 PV accumulation."""
import triton
import triton.language as tl
from triton.language.extra.corex import libdevice

@triton.jit
def quantize_probabilities(probability,value_scales):
    weighted=probability*value_scales[None,:]
    peak=tl.maximum(tl.max(tl.abs(weighted),axis=1),1.e-30)
    scale=tl.div_rn(peak,127.)
    reciprocal=tl.div_rn(1.,scale)
    quantized=libdevice.nearbyint(weighted*reciprocal[:,None])
    quantized=tl.minimum(tl.maximum(quantized,0.),127.).to(tl.int8)
    return quantized,scale

@triton.jit
def quantized_pv(probability,values,value_scales):
    tl.static_assert(probability.shape[1]>=32)
    tl.static_assert(values.dtype==tl.int8)
    quantized,scale=quantize_probabilities(probability,value_scales)
    integer=tl.dot(quantized,values,out_dtype=tl.int32)
    return integer.to(tl.float32)*scale[:,None]

@triton.jit
def probe_primitive(P,V,S,Output,Integer,Quantized,Scales,M:tl.constexpr,T:tl.constexpr,D:tl.constexpr,DIAGNOSTICS:tl.constexpr):
    rows=tl.arange(0,M);tokens=tl.arange(0,T);dims=tl.arange(0,D)
    probability=tl.load(P+rows[:,None]*T+tokens[None,:])
    values=tl.load(V+tokens[:,None]*D+dims[None,:])
    value_scales=tl.load(S+tokens)
    result=quantized_pv(probability,values,value_scales)
    tl.store(Output+rows[:,None]*D+dims[None,:],result)
    if DIAGNOSTICS:
        quantized,scale=quantize_probabilities(probability,value_scales)
        integer=tl.dot(quantized,values,out_dtype=tl.int32)
        tl.store(Integer+rows[:,None]*D+dims[None,:],integer)
        tl.store(Quantized+rows[:,None]*T+tokens[None,:],quantized)
        tl.store(Scales+rows,scale)
