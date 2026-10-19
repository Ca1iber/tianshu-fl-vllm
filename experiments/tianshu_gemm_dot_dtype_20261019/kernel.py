# SPDX-License-Identifier: Apache-2.0
"""Owned NT dot-dtype experiment: BF16 public inputs/output, FP32 accumulation."""
import torch,triton
import triton.language as tl
LAST={}

@triton.jit
def _gemm(a,weight,partial,out,m:tl.constexpr,n:tl.constexpr,k:tl.constexpr,
          bm:tl.constexpr,bn:tl.constexpr,bk:tl.constexpr,splits:tl.constexpr,
          group_m:tl.constexpr,FP16_DOT:tl.constexpr):
    pid=tl.program_id(0);grid_m,grid_n=tl.cdiv(m,bm),tl.cdiv(n,bn)
    if group_m==1:pid_m,pid_n=pid//grid_n,pid%grid_n
    else:
        width=group_m*grid_n;group_id=pid//width;first_m=group_id*group_m
        group_size=tl.minimum(grid_m-first_m,group_m)
        pid_m=first_m+(pid%width)%group_size;pid_n=(pid%width)//group_size
    mi=pid_m*bm+tl.arange(0,bm);ni=pid_n*bn+tl.arange(0,bn)
    segment=tl.program_id(1);ki=segment*bk+tl.arange(0,bk)
    ap=a+mi[:,None]*k+ki[None,:];bp=weight+ni[None,:]*k+ki[:,None]
    acc=tl.zeros((bm,bn),tl.float32)
    for step in range(tl.cdiv(k,bk*splits)):
        av=tl.load(ap,mi[:,None]<m,other=0);bv=tl.load(bp,ni[None,:]<n,other=0)
        if FP16_DOT:av=av.to(tl.float16);bv=bv.to(tl.float16)
        acc=tl.dot(av,bv,acc,input_precision=None)
        ap+=bk*splits;bp+=bk*splits
    off=mi[:,None]*n+ni[None,:];mask=(mi[:,None]<m)&(ni[None,:]<n)
    if splits==1:tl.store(out+off,acc,mask)
    else:tl.store(partial+segment*m*n+off,acc,mask)

@triton.jit
def _reduce(partial,out,count:tl.constexpr,splits:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);acc=tl.zeros((BLOCK,),tl.float32)
    for segment in range(splits):acc+=tl.load(partial+segment*count+i,i<count,other=0)
    tl.store(out+i,acc,i<count)

def prepare(a,w,c):
    assert a.dtype==torch.bfloat16 and w.dtype in (torch.bfloat16,torch.float16)
    assert a.is_contiguous() and w.is_contiguous() and a.shape[1]==w.shape[1]
    assert a.shape[1]%(c['bk']*c['splits'])==0
    m,k=a.shape;n=w.shape[0]
    out=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    partial=torch.empty((c['splits'],m,n),device=a.device,dtype=torch.float32) if c['splits']>1 else out
    return partial,out

def mm(a,w,scratch,c,fp16_dot):
    assert fp16_dot or w.dtype==torch.bfloat16
    m,k=a.shape;n=w.shape[0];partial,out=scratch
    compiled=_gemm[(triton.cdiv(m,c['bm'])*triton.cdiv(n,c['bn']),c['splits'])](a,w,partial,out,m,n,k,c['bm'],c['bn'],c['bk'],c['splits'],c['group_m'],fp16_dot,num_warps=c['warps'],num_stages=c['stages'])
    if c['splits']>1:_reduce[(triton.cdiv(m*n,1024),)](partial,out,m*n,c['splits'],1024)
    LAST.update(FP16_DOT=bool(fp16_dot),A_dtype=str(a.dtype),W_dtype=str(w.dtype),output_dtype=str(out.dtype),num_warps=compiled.metadata.num_warps,num_stages=compiled.metadata.num_stages,n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None))
    return out
