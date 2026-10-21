# SPDX-License-Identifier: Apache-2.0
"""Actual Down BF16 mm topology, including its existing M2048 specialization."""
import hashlib, importlib.util, json
from pathlib import Path
import torch, flag_gems
BASE=Path(__file__).resolve().parent
SOURCE=BASE.parent/'tianshu_prefill_gemm_tiles_20261014/experimental_mm.py'
RECORD=json.loads((BASE/'qualified_down_route.json').read_text())
assert RECORD['enabled'] and RECORD['shape']==[2048,2048,6144]
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==RECORD['kernel_sha256']
assert hashlib.sha256(Path(RECORD['vendor_source']).read_bytes()).hexdigest()==RECORD['vendor_sha256']
spec=importlib.util.spec_from_file_location('_down_existing_M2048_BF16',SOURCE)
kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)
def qualified_mm(a,w):
    assert a.is_contiguous() and w.is_contiguous() and tuple(w.shape)==(2048,6144)
    if a.shape[0]==2048:
        out=torch.empty((2048,2048),device=a.device,dtype=a.dtype)
        return kernel.mm(a,w,(out,out),RECORD['configuration'])
    return flag_gems.mm(a,w.T)
