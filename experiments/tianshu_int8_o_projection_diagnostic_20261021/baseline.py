# SPDX-License-Identifier: Apache-2.0
"""Actual unquantized QKV/O mm with the qualified O M2048 specialization."""
import hashlib,importlib.util
from pathlib import Path
import torch,flag_gems
BASE=Path(__file__).resolve().parent
SOURCE=BASE.parent/'tianshu_prefill_gemm_tiles_20261014/experimental_mm.py'
HOOK=BASE.parent/'tianshu_prefill_gemm_tiles_20261014/hooks/sitecustomize.py'
VENDOR=Path('/usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/ops/mm.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='cb556d7a6025e50d95a87a7858154fd10e547d9c145a5a18b65133122ab8c880'
assert hashlib.sha256(VENDOR.read_bytes()).hexdigest()=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
spec=importlib.util.spec_from_file_location('_attention_projection_qualified_BF16',SOURCE)
kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)
def qualified_mm(a,w):
    assert a.is_contiguous() and w.is_contiguous() and a.shape[1]==2048
    assert tuple(w.shape) in ((2560,2048),(2048,2048))
    if a.shape[0]==2048 and w.shape[0]==2048:
        out=torch.empty((2048,2048),device=a.device,dtype=a.dtype)
        return kernel.mm(a,w,(out,out),dict(bm=256,bn=256,bk=32,splits=1,warps=16,stages=2,group_m=1))
    return flag_gems.mm(a,w.T)
