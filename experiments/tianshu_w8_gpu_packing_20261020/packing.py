# SPDX-License-Identifier: Apache-2.0
"""Static weight packing using the already verified dynamic row-quantization kernel."""
import torch

def pack_into(weight,packed,scales,kernel):
    n,k=weight.shape
    assert weight.is_cuda and weight.dtype==torch.bfloat16 and weight.stride(1)==1
    assert packed.is_cuda and packed.dtype==torch.int8 and packed.shape==weight.shape and packed.is_contiguous()
    assert scales.is_cuda and scales.dtype==torch.float32 and scales.shape==(n,)
    assert weight.device==packed.device==scales.device
    compiled=[]
    # Bounded row grids also cover the 130560-row LM head without relying on a huge grid-x limit.
    for lo in range(0,n,8192):
        hi=min(lo+8192,n)
        compiled.append(kernel.quantize(weight[lo:hi],packed[lo:hi],scales[lo:hi]))
    return compiled
