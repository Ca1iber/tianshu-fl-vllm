# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import hashlib,numpy as np
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261023_native_W4_formats_CPU_v2_tianshu');GPU=Path('/workspace/benchmark-results/20261023_native_W4_formats_GPU_tianshu')
VARIANTS=['generic_low_even','generic_high_even']+[f'awq_{order}_{zeros}' for order in ('natural','permuted') for zeros in ('zeros_packed','zeros_bf16','offsets_bf16')]

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bf16(x):
 bits=np.asarray(x,dtype=np.float32).view(np.uint32);return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
def f32(bits):return (np.asarray(bits,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def permutation(name):return [0,2,4,6,1,3,5,7] if '_permuted_' in name else list(range(8))
def pack32(q,order):
 data=q.reshape(q.shape[0],-1,8)[...,order].astype(np.uint32);return np.sum(data<<(4*np.arange(8,dtype=np.uint32)),axis=-1,dtype=np.uint32).view(np.int32)
def unpack32(packed,order):
 raw=(packed.view(np.uint32)[...,None]>>(4*np.arange(8,dtype=np.uint32)))&15;out=np.empty_like(raw);out[...,order]=raw;return out.reshape(packed.shape[0],-1).astype(np.uint8)
