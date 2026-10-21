# SPDX-License-Identifier: Apache-2.0
"""Full offline-reconstructible tail fixtures for the actual split-K graph."""
import hashlib
import torch

def run(root,plans,kernel):
    def pack(a):
        scale=a.float().abs().amax(1).clamp_min(1.e-8)/127.
        return torch.round(a.float()/scale[:,None]).clamp(-127,127).to(torch.int8),scale
    cases=[];generator=torch.Generator().manual_seed(202610211205)
    for m in (3,17):
        n,k=130,259
        ac=torch.randn(m,k,generator=generator,dtype=torch.bfloat16);wc=torch.randn(n,k,generator=generator,dtype=torch.bfloat16)*.02
        ac[-1].zero_();wc[-1].zero_()
        ac[-2].zero_();ac[-2,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
        wqc,wsc=pack(wc);storage=torch.empty((m,k+7),device='cuda',dtype=torch.bfloat16);a=storage[:,:k]
        w=torch.empty((k,n),device='cuda',dtype=torch.int8);ws=wsc.cuda();q=torch.empty((m,k),device='cuda',dtype=torch.int8);asc=torch.empty(m,device='cuda')
        out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16);integer=torch.empty((m,n),device='cuda',dtype=torch.int32)
        for index,config in enumerate(plans):
            count=config['splits']*m*n;guard=torch.full((count+16,),-777,dtype=torch.int32,device='cuda');partial=guard[:count].view(config['splits'],m,n)
            a.copy_(ac);w.copy_(wqc.T);ws.copy_(wsc)
            kernel.quantize(a,q,asc);kernel.mm(q,w,asc,ws,out,integer,config,return_int=True,partial=partial);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):kernel.quantize(a,q,asc);kernel.mm(q,w,asc,ws,out,integer,config,return_int=True,partial=partial)
            for phase in range(3):
                original=ac if phase==0 else ac*(.5 if phase==1 else -.5)
                sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.
                a.copy_(original);w.copy_((wqc*sign).T);ws.copy_(wsc*factor);out.fill_(float('nan'));integer.fill_(-777);partial.fill_(-777)
                cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                graph.replay();torch.cuda.synchronize()
                qc,sc=pack(original);reference=qc.to(torch.int64)@(wqc*sign).to(torch.int64).T
                assert torch.equal(q.cpu(),qc) and torch.equal(asc.cpu(),sc) and torch.equal(integer.cpu().to(torch.int64),reference)
                assert torch.equal(partial.sum(0,dtype=torch.int64).cpu(),reference)
                expected=(reference.float()*sc[:,None]*(wsc*factor)[None,:]).to(torch.bfloat16)
                assert torch.equal(out.cpu(),expected) and torch.equal(a.cpu(),original) and torch.equal(w.cpu(),(wqc*sign).T)
                assert bool((guard[count:]==-777).all()) and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
                path=root/f'tail_M{m}_config{index}_phase{phase}.pt'
                torch.save(dict(M=m,N=n,K=k,configuration=config,phase=phase,a=original,w=wc,actual_q=q.cpu(),actual_as=asc.cpu(),actual_wq=w.cpu(),actual_ws=ws.cpu(),partial=partial.cpu(),integer=integer.cpu(),out=out.cpu(),guard=guard[count:].cpu()),path)
                cases.append(dict(M=m,N=n,K=k,index=index,phase=phase,configuration=config,full_integer_and_BF16_exact=True,full_partial_sum_exact=True,RNG_unchanged=True,guard_passed=True,raw_file=path.name,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            del graph,partial,guard
    return dict(complete=True,actual_strided_input=True,cases=cases)
