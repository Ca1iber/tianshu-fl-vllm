# SPDX-License-Identifier: Apache-2.0
"""Packed W4/W8 storage, fused BF16 decode+BF16 dot. No native INT4 claim."""
import triton
import triton.language as tl

@triton.jit
def _matmul(A,W,S,Z,O,F,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,ASTRIDE:tl.constexpr,
            BITS:tl.constexpr,GROUP:tl.constexpr,AFFINE:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,RETURN_FLOAT:tl.constexpr):
    pid=tl.program_id(0)
    tiles_n=tl.cdiv(N,BN)
    rows=(pid//tiles_n)*BM+tl.arange(0,BM)
    cols=(pid%tiles_n)*BN+tl.arange(0,BN)
    r=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for block in range(tl.cdiv(K,BK)):
        inner=block*BK+r
        a=tl.load(A+rows[:,None]*ASTRIDE+inner[None,:],(rows[:,None]<M)&(inner[None,:]<K),other=0)
        if BITS==4:
            encoded=tl.load(W+inner[:,None]*tl.cdiv(N,2)+cols[None,:]//2,
                            (inner[:,None]<K)&(cols[None,:]<N),other=0)
            qi=((encoded>>((cols[None,:]%2)*4))&15).to(tl.float32)
        else:
            qi=tl.load(W+inner[:,None]*N+cols[None,:],(inner[:,None]<K)&(cols[None,:]<N),other=0).to(tl.int8).to(tl.float32)
        if GROUP==0:group=0
        else:group=block*BK//GROUP
        scale=tl.load(S+group*N+cols,cols<N,other=0.)
        if AFFINE:
            zp=tl.load(Z+group*N+cols,cols<N,other=0).to(tl.float32)
            qi=qi-zp[None,:]
        elif BITS==4:qi=qi-8.
        weight=qi.to(tl.bfloat16) if BITS==8 else (qi*scale[None,:]).to(tl.bfloat16)
        acc=tl.dot(a,weight,acc,out_dtype=tl.float32)
    if BITS==8:
        scale=tl.load(S+cols,cols<N,other=0.)
        acc=acc*scale[None,:]
    offsets=rows[:,None]*N+cols[None,:];valid=(rows[:,None]<M)&(cols[None,:]<N)
    if RETURN_FLOAT:tl.store(F+offsets,acc,valid)
    tl.store(O+offsets,acc,valid)

@triton.jit
def _decode(W,S,Z,O,N:tl.constexpr,K:tl.constexpr,BITS:tl.constexpr,GROUP:tl.constexpr,AFFINE:tl.constexpr,BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);n=idx//K;k=idx%K;mask=idx<N*K
    if BITS==4:
        encoded=tl.load(W+k*tl.cdiv(N,2)+n//2,mask,other=0)
        val=((encoded>>((n%2)*4))&15).to(tl.float32)
    else:val=tl.load(W+k*N+n,mask,other=0).to(tl.int8).to(tl.float32)
    if GROUP==0:group=tl.full((BLOCK,),0,tl.int32)
    else:group=k//GROUP
    scale=tl.load(S+group*N+n,mask,other=0.)
    if AFFINE:val=val-tl.load(Z+group*N+n,mask,other=0).to(tl.float32)
    elif BITS==4:val=val-8.
    tl.store(O+idx,(val*scale).to(tl.bfloat16),mask)

def matmul(a,w,scales,zero,out,float_out,config,bits=4,group=128,affine=False,return_float=False):
    m,k=a.shape;n=out.shape[1]
    assert bits in (4,8) and (bits==4 or not affine)
    assert group==0 or (group>=config['bk'] and group%config['bk']==0)
    assert a.stride(1)==1 and out.shape==(m,n) and out.is_contiguous()
    assert w.shape==(k,(n+1)//2 if bits==4 else n) and w.is_contiguous()
    assert scales.shape==((k+(group or k)-1)//(group or k),n) and scales.is_contiguous()
    if affine:assert zero.shape==scales.shape and zero.is_contiguous()
    return _matmul[(triton.cdiv(m,config['bm'])*triton.cdiv(n,config['bn']),)](a,w,scales,zero,out,float_out,m,n,k,a.stride(0),bits,group,affine,config['bm'],config['bn'],config['bk'],return_float,num_warps=config['warps'],num_stages=config['stages'])

def decode(w,scales,zero,out,bits=4,group=128,affine=False):
    n,k=out.shape
    return _decode[(triton.cdiv(n*k,512),)](w,scales,zero,out,n,k,bits,group,affine,512,num_warps=4,num_stages=1)
