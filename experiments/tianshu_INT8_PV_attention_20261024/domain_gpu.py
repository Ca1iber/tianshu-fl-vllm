# SPDX-License-Identifier: Apache-2.0
"""Actual W1/W2 primitive grid with independent scalar CPU comparison before timing."""
import hashlib
import numpy as np
import domain_cpu

def validate(helper,root,torch):
    records=[]
    for tokens,warps in ((32,1),(32,2),(64,2)):
        generator=torch.Generator(device='cpu').manual_seed(202610241000+tokens+warps)
        p=torch.rand((16,tokens),generator=generator,dtype=torch.float32);p[-1].zero_()
        v=torch.randint(-127,128,(tokens,128),generator=generator,dtype=torch.int64).to(torch.int8)
        s=torch.linspace(.0001,.03,tokens,dtype=torch.float32)
        p,v,s=p.cuda(),v.cuda(),s.cuda()
        for phase in range(2):
            if phase:p.mul_(.5);v.neg_();s.mul_(1.5)
            output=torch.full((16,128),float('nan'),device='cuda')
            integer=torch.zeros((16,128),dtype=torch.int32,device='cuda')
            quantized=torch.zeros((16,tokens),dtype=torch.int8,device='cuda')
            scale=torch.zeros(16,device='cuda')
            helper.probe_primitive[(1,)](p,v,s,output,integer,quantized,scale,16,tokens,128,True,num_warps=warps,num_stages=1)
            torch.cuda.synchronize()
            arrays={k:t.cpu().numpy().copy() for k,t in
                dict(P=p,V=v,S=s,Output=output,Integer=integer,Q=quantized,Scale=scale).items()}
            counts=domain_cpu.verify_primitive(arrays)
            path=root/f'PVprimitive_T{tokens}_W{warps}_phase{phase}.npz'
            np.savez(path,**arrays)
            records.append(dict(T=tokens,warps=warps,phase=phase,file=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**counts))
    return dict(complete=True,all_3_actual_launches_and_6_phases_independent_CPU_exact=True,
        Pq_scale_INT32_and_FP32_output_words_exact=True),records
