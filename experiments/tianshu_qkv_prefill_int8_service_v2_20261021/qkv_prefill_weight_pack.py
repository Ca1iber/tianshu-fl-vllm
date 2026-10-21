# SPDX-License-Identifier: Apache-2.0
"""Declared static CPU quantization; no checkpoint or original weight mutation."""
import hashlib

def tensor_sha(tensor):
    import torch
    assert tensor.device.type=='cpu'
    return hashlib.sha256(memoryview(tensor.detach().contiguous().view(torch.uint8).numpy())).hexdigest()

def pack(weight):
    import torch
    assert weight.device.type=='cpu' and weight.dtype==torch.bfloat16 and tuple(weight.shape)==(2560,2048)
    assert bool(torch.isfinite(weight).all())
    values=weight.float();scales=values.abs().amax(1).clamp_min(1.e-8)/127.
    quantized=torch.round(values/scales[:,None]).clamp(-127,127).to(torch.int8)
    return quantized,scales

def describe(weight):
    q,s=pack(weight)
    return dict(shape=list(weight.shape),original_BF16_sha256=tensor_sha(weight),packed_logical_NK_sha256=tensor_sha(q),
        packed_physical_KN_sha256=tensor_sha(q.T.contiguous()),scales_sha256=tensor_sha(s),payload_bytes=q.numel()+s.numel()*4)
