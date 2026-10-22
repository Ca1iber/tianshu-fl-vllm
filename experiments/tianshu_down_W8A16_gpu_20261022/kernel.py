# SPDX-License-Identifier: Apache-2.0
"""W8A16 M64: BF16 activations, static int8 weight codes, FP32 scale."""
import torch,triton
import triton.language as tl

@triton.jit
def _w8a16_down_m64_v1(H,W,WS,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,NN:tl.constexpr):
 pid=tl.program_id(0);tiles_n=tl.cdiv(N,BN)
 rows=(pid//tiles_n)*BM+tl.arange(0,BM);cols=(pid%tiles_n)*BN+tl.arange(0,BN);inner=tl.arange(0,BK)
 total=tl.zeros((BM,BN),tl.float32)
 for part in range(tl.cdiv(K,BK)):
  absolute=part*BK+inner
  left=tl.load(H+rows[:,None]*K+absolute[None,:],(rows[:,None]<M)&(absolute[None,:]<K),other=0)
  if NN:right=tl.load(W+absolute[:,None]*N+cols[None,:],(cols[None,:]<N)&(absolute[:,None]<K),other=0)
  else:right=tl.load(W+cols[None,:]*K+absolute[:,None],(cols[None,:]<N)&(absolute[:,None]<K),other=0)
  total=tl.dot(left,right.to(tl.bfloat16),total,out_dtype=tl.float32)
 scales=tl.load(WS+cols,cols<N,other=0.)
 tl.store(O+rows[:,None]*N+cols[None,:],total*scales[None,:],(rows[:,None]<M)&(cols[None,:]<N))

def mm(h,w,scales,out,configuration):
 m,k=h.shape;n=2048;layout=configuration['layout'];bm,bn,bk,warps=(configuration[k] for k in ('bm','bn','bk','warps'))
 assert m==64 and k==6144 and h.dtype==out.dtype==torch.bfloat16 and w.dtype==torch.int8 and scales.dtype==torch.float32
 assert w.shape==((k,n) if layout=='NN' else (n,k)) and out.shape==(m,n) and scales.shape==(n,) and all(t.is_contiguous() for t in (h,w,scales,out))
 assert layout in ('NN','NT') and (bm,bn) in ((32,128),(64,64)) and bk in (64,128) and warps in (2,4)
 return _w8a16_down_m64_v1[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](h,w,scales,out,m,n,k,bm,bn,bk,layout=='NN',num_warps=warps,num_stages=1,enable_fp_fusion=False)
