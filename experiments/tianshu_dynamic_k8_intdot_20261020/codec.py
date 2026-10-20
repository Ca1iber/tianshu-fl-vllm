# SPDX-License-Identifier: Apache-2.0
"""CPU format contract for finite BF16 keys; per-token/head FP32 scale."""
import torch
PAGE_BYTES=12416
K_BYTES=4096
V_OFFSET=4096
SCALE_OFFSET=12288
SCALE_FLOOR=1.e-8

def encode_cpu(x):
    assert x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.shape[-1]==128
    assert bool(torch.isfinite(x).all())
    maximum=x.float().abs().amax(-1).clamp(min=SCALE_FLOOR)
    scale=maximum/127
    codes=torch.round(x.float()/scale[...,None]).clamp(-127,127).to(torch.int8)
    return codes.view(torch.uint8),scale

def decode_cpu(codes,scale):
    assert codes.device.type==scale.device.type=='cpu' and codes.dtype==torch.uint8 and scale.dtype==torch.float32
    assert codes.shape[:-1]==scale.shape and codes.shape[-1]==128
    assert bool((codes!=128).all()) and bool(torch.isfinite(scale).all()) and bool((scale>0).all())
    return (codes.view(torch.int8).float()*scale[...,None]).to(torch.bfloat16)
