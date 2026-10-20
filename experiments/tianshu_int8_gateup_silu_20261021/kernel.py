# SPDX-License-Identifier: Apache-2.0
"""Integer Gate/Up projections, original BF16 round, exact native SiLU scalar."""
import torch,triton
import triton.language as tl
from flag_gems.fused.silu_and_mul import silu_and_mul_kernel as native_activation
_native_silu_scalar=native_activation._scalar_fn

@triton.jit
def _fused(A,W,AS,WS,O,Integer,Projected,M:tl.constexpr,I:tl.constexpr,K:tl.constexpr,
           BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,DIAGNOSTICS:tl.constexpr):
    pid=tl.program_id(0);tiles_n=tl.cdiv(I,BN)
    rows=(pid//tiles_n)*BM+tl.arange(0,BM)
    columns=(pid%tiles_n)*BN+tl.arange(0,BN)
    inner=tl.arange(0,BK)
    gate_acc=tl.zeros((BM,BN),tl.int32);up_acc=tl.zeros((BM,BN),tl.int32)
    for block in range(tl.cdiv(K,BK)):
        absolute=block*BK+inner
        left=tl.load(A+rows[:,None]*K+absolute[None,:],(rows[:,None]<M)&(absolute[None,:]<K),other=0)
        gate_w=tl.load(W+columns[None,:]*K+absolute[:,None],(columns[None,:]<I)&(absolute[:,None]<K),other=0)
        up_w=tl.load(W+(columns[None,:]+I)*K+absolute[:,None],(columns[None,:]<I)&(absolute[:,None]<K),other=0)
        gate_acc=tl.dot(left,gate_w,gate_acc,out_dtype=tl.int32)
        up_acc=tl.dot(left,up_w,up_acc,out_dtype=tl.int32)
    scale_a=tl.load(AS+rows,rows<M,other=0.)
    gate_scale=tl.load(WS+columns,columns<I,other=0.)
    up_scale=tl.load(WS+I+columns,columns<I,other=0.)
    gate=(gate_acc.to(tl.float32)*scale_a[:,None]*gate_scale[None,:]).to(tl.bfloat16)
    up=(up_acc.to(tl.float32)*scale_a[:,None]*up_scale[None,:]).to(tl.bfloat16)
    activated=_native_silu_scalar(gate,up)
    mask=(rows[:,None]<M)&(columns[None,:]<I)
    tl.store(O+rows[:,None]*I+columns[None,:],activated,mask)
    if DIAGNOSTICS:
        offset=rows[:,None]*(2*I)+columns[None,:]
        tl.store(Integer+offset,gate_acc,mask);tl.store(Integer+offset+I,up_acc,mask)
        tl.store(Projected+offset,gate,mask);tl.store(Projected+offset+I,up,mask)

def fused(q,w,ascale,wscale,out,integer,projected,config,diagnostics=False):
    m,k=q.shape;n=w.shape[0];i=n//2
    assert m in (32,64) and (n,k)==(12288,2048) and i==6144
    assert q.dtype==w.dtype==torch.int8 and q.is_contiguous() and w.is_contiguous() and w.shape[1]==k
    assert ascale.dtype==wscale.dtype==torch.float32 and ascale.shape==(m,) and wscale.shape==(n,)
    assert out.dtype==torch.bfloat16 and out.shape==(m,i) and out.is_contiguous()
    assert all(t.data_ptr()%256==0 for t in (q,w,ascale,wscale,out,integer,projected))
    assert config['bk']==128 and config['bm'] in (32,64) and m%config['bm']==0
    if diagnostics:
        assert integer.dtype==torch.int32 and integer.shape==(m,n) and integer.is_contiguous()
        assert projected.dtype==torch.bfloat16 and projected.shape==(m,n) and projected.is_contiguous()
    return _fused[(triton.cdiv(m,config['bm'])*triton.cdiv(i,config['bn']),)](q,w,ascale,wscale,out,integer,projected,
        m,i,k,config['bm'],config['bn'],config['bk'],diagnostics,num_warps=config['warps'],num_stages=config['stages'])
