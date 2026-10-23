# SPDX-License-Identifier: Apache-2.0
import torch
from generated_argmax import argmax_kernel_inner

def launch(probs,noise):
    assert probs.is_cuda and probs.dtype==noise.dtype==torch.float32 and probs.ndim==2
    assert probs.shape==noise.shape and probs.is_contiguous() and noise.is_contiguous() and probs.device==noise.device
    b,n=probs.shape
    out=torch.empty((b,),device=probs.device,dtype=torch.int64)
    argmax_kernel_inner[(b,1,1)](inp=probs,noise=noise,out_index=out,M=b,N=n)
    return out
