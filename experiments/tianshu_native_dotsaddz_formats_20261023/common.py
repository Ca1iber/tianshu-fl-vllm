# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import hashlib,json,numpy as np
BASE=Path(__file__).resolve().parent;CPU=Path('/workspace/benchmark-results/20261023_native_dotsaddz_formats_CPU_tianshu');GPU=Path('/workspace/benchmark-results/20261023_native_dotsaddz_formats_GPU_tianshu');OLD=BASE.parent/'tianshu_native_W4_formats_20261023';LOGICAL=Path('/workspace/benchmark-results/20261023_native_W4_formats_CPU_v2_tianshu/logical.npz')
LAYOUTS=['n_natural','n_permuted','kn_natural','kn_permuted','nk_natural','nk_permuted'];SEMANTICS=['unsigned_add','center_add','signed_add','unsigned_subtract','signed_subtract'];VARIANTS=[a+'__'+b for a in LAYOUTS for b in SEMANTICS]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bf16(x):
 b=np.asarray(x,np.float32).view(np.uint32);return ((b+32767+((b>>16)&1))>>16).astype(np.uint16)
def pack(q,order):
 q=q.reshape(q.shape[0],-1,8)[...,order].astype(np.uint32);return np.sum(q<<(4*np.arange(8,dtype=np.uint32)),axis=-1,dtype=np.uint32).view(np.int32)
def unpack(p,order):
 v=(p.view(np.uint32)[...,None]>>(4*np.arange(8,dtype=np.uint32)))&15;o=np.empty_like(v);o[...,order]=v;return o.reshape(p.shape[0],-1).astype(np.uint8)
