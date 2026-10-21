# SPDX-License-Identifier: Apache-2.0
"""Exact INT32 split-K partials, then deterministic integer reduction."""
import importlib.util
from pathlib import Path
from partition import chunk_size
import torch,triton
import triton.language as tl
PARENT=Path(__file__).resolve().parent.parent/'tianshu_int8_down_grouped_20261021/kernel.py'
spec=importlib.util.spec_from_file_location('_down_splitk_original_integer_kernel',PARENT)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
quantize=parent.quantize
@triton.jit
def _partial(A,W,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GROUP_M:tl.constexpr,SPLITS:tl.constexpr,CHUNK:tl.constexpr):
    pid=tl.program_id(0);split=tl.program_id(1)
    tiles_n=tl.cdiv(N,BN);tiles_m=tl.cdiv(M,BM)
    per_group=GROUP_M*tiles_n;group=pid//per_group
    first=group*GROUP_M;size=tl.minimum(tiles_m-first,GROUP_M);local=pid%per_group
    rows=(first+local%size)*BM+tl.arange(0,BM)
    cols=(local//size)*BN+tl.arange(0,BN)
    inner=tl.arange(0,BK);chunk:tl.constexpr=CHUNK
    acc=tl.zeros((BM,BN),tl.int32)
    for block in range(chunk//BK):
        absolute=split*chunk+block*BK+inner
        left=tl.load(A+rows[:,None]*K+absolute[None,:],(rows[:,None]<M)&(absolute[None,:]<K),other=0)
        right=tl.load(W+absolute[:,None]*N+cols[None,:],(cols[None,:]<N)&(absolute[:,None]<K),other=0)
        acc=tl.dot(left,right,acc,out_dtype=tl.int32)
    tl.store(P+split*M*N+rows[:,None]*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))
@triton.jit
def _reduce(P,AS,WS,O,Integer,M:tl.constexpr,N:tl.constexpr,SPLITS:tl.constexpr,RETURN_INT:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=index<M*N
    acc=tl.full((BLOCK,),0,tl.int32)
    for split in tl.static_range(SPLITS):acc+=tl.load(P+split*M*N+index,mask,other=0)
    if RETURN_INT:tl.store(Integer+index,acc,mask)
    sa=tl.load(AS+index//N,mask,other=0.);sw=tl.load(WS+index%N,mask,other=0.)
    tl.store(O+index,acc.to(tl.float32)*sa*sw,mask)

def mm(q,w,ascale,wscale,out,integer,config,return_int=False,*,partial):
    m,k=q.shape;n=w.shape[1];splits=config['splits']
    assert q.dtype==w.dtype==torch.int8 and q.is_contiguous() and w.is_contiguous() and w.shape[0]==k
    assert ascale.dtype==wscale.dtype==torch.float32 and ascale.shape==(m,) and wscale.shape==(n,)
    assert out.shape==(m,n) and out.dtype==torch.bfloat16 and out.is_contiguous()
    assert partial.dtype==torch.int32 and partial.shape==(splits,m,n) and partial.is_contiguous()
    assert not return_int or (integer.shape==(m,n) and integer.dtype==torch.int32 and integer.is_contiguous())
    assert k*127*127<2**31 and 1<=splits<=8
    first=_partial[(triton.cdiv(m,config['bm'])*triton.cdiv(n,config['bn']),splits)](q,w,partial,m,n,k,config['bm'],config['bn'],config['bk'],config['group_m'],splits,chunk_size(k,splits,config['bk']),num_warps=config['warps'],num_stages=config['stages'])
    second=_reduce[(triton.cdiv(m*n,1024),)](partial,ascale,wscale,out,integer,m,n,splits,return_int,1024,num_warps=4,num_stages=1)
    return first,second

def unsplit(q,w,ascale,wscale,out,integer):
    config=dict(bm=128,bn=128,bk=128,warps=4,stages=1,group_m=1)
    return parent.mm(q,w,ascale,wscale,out,integer,config)
