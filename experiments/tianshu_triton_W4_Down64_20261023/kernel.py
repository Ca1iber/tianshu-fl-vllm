# SPDX-License-Identifier: Apache-2.0
"""M64 W4A16: natural-N uint4, group128 BF16 scales/offset, BF16 dot inputs."""
import triton
import triton.language as tl

@triton.jit
def _mm(X,W,S,Z,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,PACKED:tl.constexpr,SYMMETRIC:tl.constexpr):
    pid=tl.program_id(0);tiles_n=tl.cdiv(N,BN)
    rows=(pid//tiles_n)*BM+tl.arange(0,BM)
    columns=(pid%tiles_n)*BN+tl.arange(0,BN)
    inner=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for group in range(K//128):
        scale=tl.load(S+group*N+columns,columns<N,other=0).to(tl.float32)
        if not SYMMETRIC:zero=tl.load(Z+group*N+columns,columns<N,other=0).to(tl.float32)
        for part in tl.static_range(128//BK):
            absolute=group*128+part*BK+inner
            left=tl.load(X+rows[:,None]*K+absolute[None,:],rows[:,None]<M,other=0)
            if PACKED:
                words=(pid%tiles_n)*(BN//8)+tl.arange(0,BN//8)
                packed=tl.load(W+absolute[:,None]*(N//8)+words[None,:],words[None,:]<(N//8),other=0).to(tl.uint32)
                q=((packed[:,:,None]>>(4*tl.arange(0,8)[None,None,:]))&15).reshape((BK,BN))
            else:
                packed=tl.load(W+absolute[:,None]*(N//8)+(columns[None,:]//8),columns[None,:]<N,other=0).to(tl.uint32)
                q=(packed>>(4*(columns[None,:]%8)))&15
            if SYMMETRIC:right=((q.to(tl.int32)-8).to(tl.float32)*scale[None,:]).to(tl.bfloat16)
            else:right=(q.to(tl.float32)*scale[None,:]+zero[None,:]).to(tl.bfloat16)
            acc=tl.dot(left,right,acc)
    tl.store(O+rows[:,None]*N+columns[None,:],acc,(rows[:,None]<M)&(columns[None,:]<N))

def run(x,w,s,z,out,cfg):
    m,k=x.shape;n=out.shape[1]
    return _mm[(triton.cdiv(m,cfg['bm'])*triton.cdiv(n,cfg['bn']),)](x,w,s,z,out,m,n,k,cfg['bm'],cfg['bn'],cfg['bk'],cfg['packed'],cfg['symmetric'],num_warps=cfg['warps'],num_stages=cfg['stages'])
