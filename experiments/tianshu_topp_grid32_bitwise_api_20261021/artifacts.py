# SPDX-License-Identifier: Apache-2.0
"""Lossless input plus finite masks, full output hashes and actual FP32 p vectors."""
import hashlib
from pathlib import Path
import numpy as np

def tensor_sha(tensor,torch):
    return hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy())).hexdigest()

def save_input(root,case,values,torch):
    path=Path(root)/f'input_{case}.pt';cpu=values.cpu().contiguous()
    record=dict(values=cpu,shape=list(cpu.shape),original_stride=list(values.stride()),sha256=tensor_sha(cpu,torch))
    torch.save(record,path)
    return dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),tensor_sha256=record['sha256'],shape=record['shape'],original_stride=record['original_stride'])

def save_outputs(root,case,pindex,phase,working,p,torch):
    record=dict(case=case,pindex=pindex,phase=phase,p=p.cpu(),outputs={})
    for name,tensor in working.items():
        cpu=tensor.cpu().contiguous();finite=torch.isfinite(cpu)
        packed=np.packbits(finite.numpy().reshape(-1),bitorder='little')
        record['outputs'][name]=dict(shape=list(cpu.shape),packed_finite=torch.from_numpy(packed),full_FP32_sha256=tensor_sha(cpu,torch),actual_stride=list(tensor.stride()))
    path=Path(root)/f'mask_{case}_{pindex}_{phase}.pt';torch.save(record,path)
    return dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
