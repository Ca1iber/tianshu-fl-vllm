# SPDX-License-Identifier: Apache-2.0
"""Batched independent framework reference, never in timed CUDA Graphs."""
import torch

def scores(q,keys,lengths):
    n=q.shape[0];qg=q.float().reshape(n,2,8,128);kg=keys.float().permute(0,2,3,1)
    result=torch.matmul(qg,kg)*(128**-.5)
    positions=torch.arange(keys.shape[1],device=q.device)
    valid=positions[None,:]<torch.tensor(lengths,device=q.device)[:,None]
    return result.masked_fill(~valid[:,None,None,:],-torch.inf)

def ideal(score,values):
    answer=torch.matmul(score.softmax(-1),values.float().permute(0,2,1,3))
    return answer.reshape(score.shape[0],16,128)

def integer_pv(score,vq,vs,lengths,tile,segments):
    n,kv,heads,width=score.shape;device=score.device
    lens=torch.tensor(lengths,device=device,dtype=torch.int64)
    divisor=torch.where((n>24)&(lens<=8192),4,segments)
    per=torch.div(lens+divisor*tile-1,divisor*tile,rounding_mode='floor')
    max_per=max((length+(4 if n>24 and length<=8192 else segments)*tile-1)//((4 if n>24 and length<=8192 else segments)*tile) for length in lengths)
    values=vq.float().permute(0,2,1,3);scales=vs.permute(0,2,1)
    offsets=torch.arange(tile,device=device)
    maxima=[];totals=[];accs=[]
    for segment in range(segments):
        m=torch.full((n,2,8),-torch.inf,device=device);total=torch.zeros_like(m);acc=torch.zeros((n,2,8,128),device=device)
        for step in range(max_per):
            pos=(segment*per+step)[:,None]*tile+offsets[None,:]
            valid=(step<per)[:,None]&(pos<lens[:,None]);safe=pos.clamp(max=width-1)
            s=torch.gather(score,3,safe[:,None,None,:].expand(n,2,8,tile))
            s=s.masked_fill(~valid[:,None,None,:],-torch.inf)
            mnew=torch.maximum(m,s.max(-1).values);mnew=torch.where(torch.isfinite(mnew),mnew,0.)
            p=torch.exp(s-mnew[...,None]);alpha=torch.exp(m-mnew)
            value_scales=torch.gather(scales,2,safe[:,None,:].expand(n,2,tile))
            weighted=p*value_scales[:,:,None,:]
            ps=weighted.abs().amax(-1).clamp_min(1.e-30)/127.
            pq=(weighted/ps[...,None]).round().clamp(0,127).to(torch.int8)
            v=torch.gather(values,2,safe[:,None,:,None].expand(n,2,tile,128))
            integer=torch.matmul(pq.float(),v)
            acc=acc*alpha[...,None]+integer*ps[...,None]
            total=total*alpha+p.sum(-1);m=mnew
        active=(segment*per*tile<lens)[:,None,None]
        maxima.append(torch.where(active,m,-torch.inf));totals.append(torch.where(active,total,0.));accs.append(torch.where(active[...,None],acc,0.))
    maxima=torch.stack(maxima);totals=torch.stack(totals);accs=torch.stack(accs)
    weight=torch.exp(maxima-maxima.max(0).values)
    denominator=(totals*weight).sum(0)
    answer=(accs*weight[...,None]).sum(0)/denominator[...,None]
    return answer.reshape(n,16,128)
