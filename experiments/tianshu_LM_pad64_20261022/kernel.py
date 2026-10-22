# SPDX-License-Identifier: Apache-2.0
"""Per-token dynamic A8, per-output static W8 and integer NT GEMM."""
import torch
import triton
import triton.language as tl
from triton.language.extra.corex import libdevice

@triton.jit
def _quantize(A,Q,S,M:tl.constexpr,K:tl.constexpr,ASTRIDE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    offsets=tl.arange(0,BLOCK)
    value=tl.load(A+row*ASTRIDE+offsets,offsets<K,other=0).to(tl.float32)
    scale=tl.div_rn(tl.maximum(tl.max(tl.abs(value)),1.e-8),127.)
    packed=libdevice.nearbyint(tl.div_rn(value,scale))
    packed=tl.minimum(tl.maximum(packed,-127.),127.).to(tl.int8)
    tl.store(Q+row*K+offsets,packed,offsets<K)
    tl.store(S+row,scale)

@triton.jit
def _gemm(A,W,AS,WS,O,Integer,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,RETURN_INT:tl.constexpr):
    pid=tl.program_id(0)
    tiles_n=tl.cdiv(N,BN)
    rows=(pid//tiles_n)*BM+tl.arange(0,BM)
    columns=(pid%tiles_n)*BN+tl.arange(0,BN)
    inner=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.int32)
    for block in range(tl.cdiv(K,BK)):
        absolute=block*BK+inner
        left=tl.load(A+rows[:,None]*K+absolute[None,:],(rows[:,None]<M)&(absolute[None,:]<K),other=0)
        right=tl.load(W+columns[None,:]*K+absolute[:,None],(columns[None,:]<N)&(absolute[:,None]<K),other=0)
        acc=tl.dot(left,right,acc,out_dtype=tl.int32)
    offset=rows[:,None]*N+columns[None,:]
    mask=(rows[:,None]<M)&(columns[None,:]<N)
    if RETURN_INT:tl.store(Integer+offset,acc,mask)
    scale_a=tl.load(AS+rows,rows<M,other=0.)
    scale_w=tl.load(WS+columns,columns<N,other=0.)
    value=acc.to(tl.float32)*scale_a[:,None]*scale_w[None,:]
    tl.store(O+offset,value.to(tl.bfloat16).to(tl.float32),mask)

def quantize(a,q,scales):
    m,k=a.shape
    assert a.dtype==torch.bfloat16 and q.dtype==torch.int8 and scales.dtype==torch.float32
    assert a.stride(1)==1 and q.shape==a.shape and q.is_contiguous() and scales.shape==(m,)
    return _quantize[(m,)](a,q,scales,m,k,a.stride(0),triton.next_power_of_2(k),num_warps=4,num_stages=1)

def mm(q,w,ascale,wscale,out,integer,config,return_int=False):
    m,k=q.shape;n=w.shape[0]
    assert q.dtype==w.dtype==torch.int8 and q.is_contiguous() and w.is_contiguous() and w.shape[1]==k
    assert ascale.dtype==wscale.dtype==torch.float32 and ascale.shape==(m,) and wscale.shape==(n,)
    assert out.shape==(m,n) and out.dtype==torch.float32 and out.is_contiguous()
    assert not return_int or (integer.shape==(m,n) and integer.dtype==torch.int32 and integer.is_contiguous())
    return _gemm[(triton.cdiv(m,config['bm'])*triton.cdiv(n,config['bn']),)](q,w,ascale,wscale,out,integer,
        m,n,k,config['bm'],config['bn'],config['bk'],return_int,num_warps=config['warps'],num_stages=config['stages'])

@triton.jit
def _native_dequant(Integer,AS,WS,O,M:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
    offset=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=offset<M*N
    acc=tl.load(Integer+offset,mask,other=0).to(tl.float32)
    asc=tl.load(AS+offset//N,mask,other=0.);ws=tl.load(WS+offset%N,mask,other=0.)
    value=acc*asc*ws
    tl.store(O+offset,value.to(tl.bfloat16).to(tl.float32),mask)
def native_dequant(integer,asc,ws,out,block):
    m,n=integer.shape
    assert out.dtype==torch.float32 and out.shape==(m,n) and out.is_contiguous() and integer.dtype==torch.int32
    return _native_dequant[(triton.cdiv(m*n,block),)](integer,asc,ws,out,m,n,block,num_warps=4,num_stages=1)

@triton.jit(do_not_specialize=['M'])
def _quantize_padded(A,Q,S,M,STRIDE:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK)
    value=tl.load(A+row*STRIDE+col,(row<M)&(col<K),other=0.).to(tl.float32)
    scale=tl.div_rn(tl.maximum(tl.max(tl.abs(value)),1.e-8),127.)
    code=libdevice.nearbyint(tl.div_rn(value,scale));code=tl.minimum(tl.maximum(code,-127.),127.).to(tl.int8)
    tl.store(Q+row*K+col,code,col<K);tl.store(S+row,scale)
def quantize_padded(a,q,s):
    m,k=a.shape
    assert 1<=m<=64 and k==2048 and a.dtype==torch.bfloat16 and a.stride(1)==1 and q.shape==(64,k) and s.shape==(64,)
    return _quantize_padded[(64,)](a,q,s,m,a.stride(0),k,2048,num_warps=4,num_stages=1)
