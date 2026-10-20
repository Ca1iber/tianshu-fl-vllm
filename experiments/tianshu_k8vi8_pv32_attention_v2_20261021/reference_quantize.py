# SPDX-License-Identifier: Apache-2.0
"""Verified explicit FP32 RNE only for reference fixtures, never timed writer."""
import hashlib,importlib.util
from pathlib import Path
import torch
KERNEL=Path(__file__).resolve().parent.parent/'tianshu_dense_w8a8_20261020/kernel.py'
assert hashlib.sha256(KERNEL.read_bytes()).hexdigest()=='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152'
spec=importlib.util.spec_from_file_location('_kvi8_verified_reference_RNE',KERNEL)
kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)

def quantize(x):
    assert x.dtype==torch.bfloat16 and x.is_cuda and x.shape[-1]==128 and x.is_contiguous()
    a=x.view(-1,128);q=torch.empty(a.shape,device=x.device,dtype=torch.int8);s=torch.empty(a.shape[0],device=x.device,dtype=torch.float32)
    kernel.quantize(a,q,s)
    return q.view(x.shape),s.view(x.shape[:-1])
