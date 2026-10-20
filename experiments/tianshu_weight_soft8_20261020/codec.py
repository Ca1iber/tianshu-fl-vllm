# SPDX-License-Identifier: Apache-2.0
"""Derived software-byte weights: per-output-row exponent origin, BF16 decode.

Three fraction bits, 1 sign bit; codes1..7 flush to signed zero. This is not
native hardware FP8. Checkpoints remain BF16; any model-quality change is unvalidated.
"""
import math
import struct
BASES = tuple(range(800, 961, 16))

def bf16_value(bits):
    return struct.unpack('>f',struct.pack('>I',bits<<16))[0]

def decode_bits(code, base):
    assert base in BASES and 0 <= code < 256
    magnitude=code&127
    return (((magnitude+base)<<4) if magnitude>=8 else 0) | ((code&128)<<8)

def encode_bits(bits, base):
    assert base in BASES and 0 <= bits < 65536
    magnitude=bits&32767
    rounded=((magnitude+7+((magnitude>>4)&1))>>4)-base
    return (min(127,rounded) if rounded>=8 else 0) | ((bits&32768)>>8)

def independent_reference(bits, base):
    """Round a scalar exact BF16 value in real-number space, then flush/clamp."""
    sign=bits&32768
    value=abs(bf16_value(bits))
    lower=bf16_value((base+8)<<4)
    upper=bf16_value((base+127)<<4)
    if not math.isfinite(value):
        return ((base+127)<<4)|sign
    if value==0:
        return sign
    _,exponent=math.frexp(value)
    step=math.ldexp(1.,exponent-4)
    value=round(value/step)*step
    value=0. if value<lower else min(upper,value)
    rounded=struct.unpack('>I',struct.pack('>f',value))[0]>>16
    return rounded|sign

def prepare_cpu(weight):
    import torch
    assert weight.device.type=='cpu' and weight.dtype==torch.bfloat16 and weight.ndim==2
    assert weight.is_contiguous() and bool(weight.isfinite().all())
    maxima=weight.float().abs().amax(dim=1)
    bases=torch.full((weight.shape[0],),-1,dtype=torch.int16)
    for base in BASES:
        limit=bf16_value((base+127)<<4)
        bases[(bases<0)&(maxima<=limit)]=base
    assert bool((bases>=0).all()),'Row maximum outside supported codec range'
    bits=weight.view(torch.int16).to(torch.int32)
    magnitude=bits&32767
    rounded=((magnitude+7+((magnitude>>4)&1))>>4)-bases[:,None].to(torch.int32)
    assert not bool((rounded>127).any()),'Unexpected saturation'
    codes=(torch.where(rounded>=8,rounded.clamp(max=127),0)|((bits&32768)>>8)).to(torch.uint8)
    decoded=decode_cpu(codes,bases)
    delta=decoded.float()-weight.float()
    return codes,bases,dict(weight_max_abs=float(maxima.max()),quantization_max_abs=float(delta.abs().max()),
                           relative_L2=float(delta.norm()/weight.float().norm()),
                           underflow_elements=int(((codes&127)==0).sum()),saturated_elements=0,
                           row_base_counts={str(base):int((bases==base).sum()) for base in BASES})

def decode_cpu(codes,bases):
    import torch
    assert codes.device.type==bases.device.type=='cpu' and codes.dtype==torch.uint8 and bases.dtype==torch.int16
    values=codes.to(torch.int32)
    magnitude=values&127
    bits=torch.where(magnitude>=8,(magnitude+bases[:,None].to(torch.int32))<<4,0)|((values&128)<<8)
    return bits.to(torch.int16).view(torch.bfloat16)
