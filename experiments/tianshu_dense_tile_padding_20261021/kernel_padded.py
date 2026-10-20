# SPDX-License-Identifier: Apache-2.0
"""Quantize real rows and zero padded rows inside the measured GPU kernel."""
import hashlib,importlib.util
from pathlib import Path
import torch,triton
import triton.language as tl
from triton.language.extra.corex import libdevice
PARENT=Path(__file__).resolve().parent.parent/'tianshu_dense_w8a8_20261020/kernel.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152'
spec=importlib.util.spec_from_file_location('_tile_pad_qualified_original_integer',PARENT)
_original=importlib.util.module_from_spec(spec);spec.loader.exec_module(_original)

@triton.jit(do_not_specialize=['REAL_M'])
def _quantize_padded(A,Q,S,REAL_M,K:tl.constexpr,ASTRIDE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    offsets=tl.arange(0,BLOCK)
    value=tl.load(A+row*ASTRIDE+offsets,(row<REAL_M)&(offsets<K),other=0).to(tl.float32)
    scale=tl.div_rn(tl.maximum(tl.max(tl.abs(value)),1.e-8),127.)
    packed=libdevice.nearbyint(tl.div_rn(value,scale))
    packed=tl.minimum(tl.maximum(packed,-127.),127.).to(tl.int8)
    tl.store(Q+row*K+offsets,packed,offsets<K)
    tl.store(S+row,scale)

def quantize(a,q,scales):
    m,k=a.shape;pad=q.shape[0]
    assert 1<=m<=pad and pad in (32,64) and q.shape==(pad,k) and scales.shape==(pad,)
    assert a.dtype==torch.bfloat16 and a.stride(1)==1 and q.dtype==torch.int8 and q.is_contiguous() and scales.dtype==torch.float32
    assert q.data_ptr()%256==scales.data_ptr()%256==0
    if m==pad:return _original.quantize(a,q,scales)
    return _quantize_padded[(pad,)](a,q,scales,m,k,a.stride(0),triton.next_power_of_2(k),num_warps=4,num_stages=1)

def mm(q,w,ascale,wscale,out,integer,config,return_int=False):
    assert q.shape[0] in (32,64) and config['bm']==q.shape[0]
    assert all(t.data_ptr()%256==0 for t in (q,w,ascale,wscale,out,integer))
    return _original.mm(q,w,ascale,wscale,out,integer,config,return_int=return_int)
