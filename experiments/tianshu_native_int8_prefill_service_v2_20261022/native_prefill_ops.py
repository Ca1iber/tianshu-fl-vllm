# SPDX-License-Identifier: Apache-2.0
import torch
_CONTEXT=None
def configure(context):
    global _CONTEXT
    assert _CONTEXT is None
    _CONTEXT=context
@torch.library.custom_op('tianshu_native_prefill::fused_or_original',mutates_args=(),device_types='cuda')
def fused_or_original(x:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
    assert _CONTEXT is not None
    return _CONTEXT(x,weight,packed,scales,bank_id)
@fused_or_original.register_fake
def fake(x,weight,packed,scales,bank_id):
    assert x.ndim==2 and x.dtype==torch.bfloat16 and x.shape[1]==2048
    return torch.empty((x.shape[0],6144),dtype=x.dtype,device=x.device)
