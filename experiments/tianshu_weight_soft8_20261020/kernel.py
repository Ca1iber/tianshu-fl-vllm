# SPDX-License-Identifier: Apache-2.0
"""NT GEMM with BF16 inputs/output, FP32 accumulation, optional software-byte W."""
import torch
import triton
import triton.language as tl
LAST={}

@triton.jit
def _decode(code,base):
    packed=code.to(tl.uint32)
    magnitude=packed&127
    bits=tl.where(magnitude>=8,(magnitude+base)<<4,0)|((packed&128)<<8)
    return bits.to(tl.uint16).to(tl.bfloat16,bitcast=True)

@triton.jit
def _gemm(a,weight,bases,partial,out,m:tl.constexpr,n:tl.constexpr,k:tl.constexpr,
          bm:tl.constexpr,bn:tl.constexpr,bk:tl.constexpr,splits:tl.constexpr,W8:tl.constexpr):
    pid=tl.program_id(0)
    grid_n=tl.cdiv(n,bn)
    mi=(pid//grid_n)*bm+tl.arange(0,bm)
    ni=(pid%grid_n)*bn+tl.arange(0,bn)
    segment=tl.program_id(1)
    ki=segment*bk+tl.arange(0,bk)
    ap=a+mi[:,None]*k+ki[None,:]
    bp=weight+ni[None,:]*k+ki[:,None]
    if W8:
        base=tl.load(bases+ni,ni<n,other=800).to(tl.uint32)
    acc=tl.zeros((bm,bn),tl.float32)
    for step in range(tl.cdiv(k,bk*splits)):
        av=tl.load(ap,mi[:,None]<m,other=0)
        bv=tl.load(bp,ni[None,:]<n,other=0)
        if W8:
            bv=_decode(bv,base[None,:])
        acc=tl.dot(av,bv,acc,input_precision=None)
        ap+=bk*splits
        bp+=bk*splits
    offset=mi[:,None]*n+ni[None,:]
    mask=(mi[:,None]<m)&(ni[None,:]<n)
    if splits==1:
        tl.store(out+offset,acc,mask)
    else:
        tl.store(partial+segment*m*n+offset,acc,mask)

@triton.jit
def _reduce(partial,out,count:tl.constexpr,splits:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    acc=tl.zeros((BLOCK,),tl.float32)
    for segment in range(splits):
        acc+=tl.load(partial+segment*count+i,i<count,other=0)
    tl.store(out+i,acc,i<count)

@triton.jit
def decode_probe(codes,bases,out,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    code=tl.load(codes+i,i<N,other=0)
    base=tl.load(bases+i,i<N,other=800).to(tl.uint32)
    tl.store(out+i,_decode(code,base),i<N)

def mm(a,w,bases,scratch,c):
    assert a.dtype==torch.bfloat16 and w.dtype in (torch.bfloat16,torch.uint8)
    assert a.is_contiguous() and w.is_contiguous() and bases.is_contiguous()
    assert a.shape[1]==w.shape[1] and a.shape[1]%(c['bk']*c['splits'])==0
    m,k=a.shape
    n=w.shape[0]
    partial,out=scratch
    assert out.shape==(m,n) and out.dtype==torch.bfloat16
    compiled=_gemm[(triton.cdiv(m,c['bm'])*triton.cdiv(n,c['bn']),c['splits'])](
        a,w,bases,partial,out,m,n,k,c['bm'],c['bn'],c['bk'],c['splits'],w.dtype==torch.uint8,
        num_warps=c['warps'],num_stages=c['stages'])
    if c['splits']>1:
        _reduce[(triton.cdiv(m*n,1024),)](partial,out,m*n,c['splits'],1024)
    LAST.update(W8=w.dtype==torch.uint8,public_A=str(a.dtype),stored_W=str(w.dtype),public_C=str(out.dtype),
                FP32_accumulation=True,num_warps=compiled.metadata.num_warps,num_stages=compiled.metadata.num_stages,
                n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None))
    return out
