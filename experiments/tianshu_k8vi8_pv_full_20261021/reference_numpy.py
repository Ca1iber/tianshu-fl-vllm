# SPDX-License-Identifier: Apache-2.0
"""Independent CPU FP64 ideal and explicit FP32/RNE/INT64 tiled-PV oracle."""
import numpy as np
import torch

def soft8(x):
    bits=x.contiguous().view(torch.int16).numpy().astype(np.uint16).astype(np.uint32)
    mag=bits&32767;rounded=(mag+7+((mag>>4)&1))>>4
    code=(np.where(rounded>=936,np.minimum(rounded-928,127),0)|((bits&32768)>>8)).astype(np.uint8)
    value=code.astype(np.uint32);magnitude=value&127
    bf16=(np.where(magnitude>=8,(magnitude<<4)+14848,0)|((value&128)<<8)).astype(np.uint32)
    return (bf16<<16).view(np.float32),code

def quantize(x):
    x=np.asarray(x,dtype=np.float32)
    scale=(np.maximum(np.max(np.abs(x),axis=-1),np.float32(1.e-8))/np.float32(127.)).astype(np.float32)
    q=np.clip(np.rint((x/scale[...,None]).astype(np.float32)),-127,127).astype(np.int8)
    return q,scale

class Oracle:
    def __init__(self,data):
        self.data=data;self.cache={}
        self.q=data['q'].float().numpy();self.k,self.kcodes=soft8(data['k'])
        self.v=data['v'].float().numpy();self.vq,self.vs=quantize(self.v)
        self.k_original=data['k'].float().numpy()
    def score(self,index,head):
        length=self.data['lengths'][index];kv=head//8
        q=self.q[index,head].astype(np.float64);keys=self.k[index,:length,kv].astype(np.float64)
        dot=keys@q;return dot,np.float32(128**-.5)
    def ideal(self,index,head,kind='v8'):
        key=('ideal',index,head,kind)
        if key in self.cache:return self.cache[key]
        length=self.data['lengths'][index];kv=head//8;q=self.q[index,head].astype(np.float64)
        keys=self.k_original[index,:length,kv] if kind=='original' else self.k[index,:length,kv]
        values=self.v[index,:length,kv] if kind in ('original','current') else self.vq[index,:length,kv].astype(np.float64)*self.vs[index,:length,kv,None]
        scores=keys.astype(np.float64)@q*(128**-.5);p=np.exp(scores-scores.max());p/=p.sum()
        result=p@values.astype(np.float64);self.cache[key]=result;return result
    def integer_pv(self,index,head,tile,segments):
        key=('integer',index,head,tile,segments)
        if key in self.cache:return self.cache[key]
        length=self.data['lengths'][index];n=self.data['batch'];kv=head//8
        dot,scale=self.score(index,head);scores=(dot.astype(np.float32)*scale).astype(np.float32)
        divisor=4 if n>24 and length<=8192 else segments
        per=(length+divisor*tile-1)//(divisor*tile)
        maxima=[];sums=[];accumulators=[];bounds=[]
        for segment in range(segments):
            start=segment*per*tile
            if start>=length:continue
            m=np.float32(-np.inf);total=np.float32(0.);acc=np.zeros(128,dtype=np.float32);bound=np.zeros(128,dtype=np.float64)
            for lo in range(start,min(length,start+per*tile),tile):
                hi=min(length,lo+tile);s=scores[lo:hi];mnew=np.maximum(m,s.max()).astype(np.float32)
                alpha=np.float32(np.exp(np.float32(m-mnew)))
                p=np.exp((s-mnew).astype(np.float32)).astype(np.float32)
                weighted=(p*self.vs[index,lo:hi,kv]).astype(np.float32)
                ps=np.float32(max(np.max(np.abs(weighted)),np.float32(1.e-30))/np.float32(127.))
                pq=np.clip(np.rint((weighted/ps).astype(np.float32)),0,127).astype(np.int64)
                integer=pq@self.vq[index,lo:hi,kv].astype(np.int64)
                assert tile*127*127<2**31 and np.max(np.abs(integer))<=tile*127*127
                acc=((acc*alpha).astype(np.float32)+(integer.astype(np.float32)*ps).astype(np.float32)).astype(np.float32)
                total=np.float32(np.float32(total*alpha)+p.sum(dtype=np.float32))
                bound=bound*float(alpha)+.501*float(ps)*np.abs(self.vq[index,lo:hi,kv].astype(np.int64)).sum(0)
                m=mnew
            maxima.append(m);sums.append(total);accumulators.append(acc);bounds.append(bound)
        maxima=np.asarray(maxima,dtype=np.float32);weights=np.exp((maxima-maxima.max()).astype(np.float32)).astype(np.float32)
        denominator=(np.asarray(sums,dtype=np.float32)*weights).sum(dtype=np.float32)
        numerator=(np.stack(accumulators)*weights[:,None]).sum(0,dtype=np.float32)
        result=(numerator/denominator).astype(np.float32)
        bound=(np.stack(bounds)*weights.astype(np.float64)[:,None]).sum(0)/float(denominator)
        self.cache[key]=(result,bound);return result,bound
