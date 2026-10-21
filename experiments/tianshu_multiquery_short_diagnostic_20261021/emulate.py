# SPDX-License-Identifier: Apache-2.0
"""Independent CPU emulation of existing online softmax + BF16(P) @ BF16(V)."""
import torch

def emulate(q,k,v,qlens,lengths,config,split,segments,causal=True):
    assert q.device.type==k.device.type==v.device.type=='cpu'
    n=len(qlens);tile=config['tile'];bq=config['block_m']//8;outputs=[];offset=0
    for seq,(ql,kl) in enumerate(zip(qlens,lengths,strict=True)):
      for pos in range(ql):
        houtputs=[]
        prefix=min(kl,kl-ql+(pos//bq+1)*bq) if causal else kl
        nt=(prefix+tile-1)//tile
        tps=(kl+segments*tile-1)//(segments*tile)
        if n>24 and kl<=8192:tps=(kl+4*tile-1)//(4*tile)
        active=(kl+tps*tile-1)//(tps*tile) if split else 1
        for h in range(16):
          score=(k[seq,:kl,h//8].float()@q[offset+pos,h].float())*(128**-.5)
          if causal:score[kl-ql+pos+1:]=-float('inf')
          vv=v[seq,:kl,h//8].float();parts=[]
          for seg in range(active):
            start=seg*tps if split else 0;end=min((seg+1)*tps,nt) if split else nt
            maximum=torch.tensor(-float('inf'));denominator=torch.tensor(0.);acc=torch.zeros(128)
            for t in range(start,end):
                lo,hi=t*tile,min((t+1)*tile,kl)
                s=score[lo:hi];m=torch.maximum(maximum,s.max());m=torch.where(torch.isfinite(m),m,torch.tensor(0.))
                p=(s-m).exp();alpha=(maximum-m).exp()
                denominator=denominator*alpha+p.sum();acc=acc*alpha+p.to(torch.bfloat16).float()@vv[lo:hi];maximum=m
            parts.append((maximum,denominator,acc))
          maximum=torch.stack([p[0] for p in parts]).max()
          factors=[(p[0]-maximum).exp() for p in parts]
          denominator=sum(p[1]*f for p,f in zip(parts,factors,strict=True))
          acc=sum(p[2]*f for p,f in zip(parts,factors,strict=True))
          houtputs.append(torch.where(denominator==0,torch.zeros_like(acc),acc/denominator))
        outputs.append(torch.stack(houtputs))
      offset+=ql
    return torch.stack(outputs).to(torch.bfloat16)
