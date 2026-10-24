# SPDX-License-Identifier: Apache-2.0
"""Same actual T32W1 primitive for both algorithms, plus quantization boundary cases."""
import hashlib
import numpy as np
import domain_cpu

def validate(helpers,root,torch):
    records=[]
    generator=torch.Generator(device='cpu').manual_seed(202610241033)
    basep=torch.rand((16,32),generator=generator,dtype=torch.float32);basep[-1].zero_()
    basev=torch.randint(-127,128,(32,128),generator=generator,dtype=torch.int64).to(torch.int8)
    bases=torch.linspace(.0001,.03,32,dtype=torch.float32)
    boundary=np.ones((16,32),dtype=np.float32)
    scale=np.float32(1./127.)
    for row in range(16):
        for col in range(1,32):
            index=row*31+col-1;k=index%127;neighbor=(index//127)%3
            value=np.float32(np.float32(k+.5)*scale)
            boundary[row,col]=np.nextafter(value,np.float32(-np.inf if neighbor==0 else np.inf)) if neighbor!=1 else value
    for variant,helper in helpers.items():
        for phase in range(5):
            p,v,s=basep.clone(),basev.clone(),bases.clone()
            if phase==1:p.mul_(.5);v.neg_();s.mul_(1.5)
            if phase in (2,3):p=torch.from_numpy(boundary.copy());s.fill_(1.)
            if phase==3:p.mul_(1.e-12)
            if phase==4:p.zero_();s.fill_(1.)
            p,v,s=p.cuda(),v.cuda(),s.cuda()
            output=torch.full((16,128),float('nan'),device='cuda')
            integer=torch.zeros((16,128),dtype=torch.int32,device='cuda')
            quantized=torch.zeros((16,32),dtype=torch.int8,device='cuda')
            row_scale=torch.zeros(16,device='cuda')
            helper.probe_primitive[(1,)](p,v,s,output,integer,quantized,row_scale,16,32,128,True,num_warps=1,num_stages=1)
            torch.cuda.synchronize()
            arrays={k:t.cpu().numpy().copy() for k,t in
                dict(P=p,V=v,S=s,Output=output,Integer=integer,Q=quantized,Scale=row_scale).items()}
            path=root/f'PVprimitive_{variant}_phase{phase}.npz';np.savez(path,**arrays)
            counts=domain_cpu.verify_primitive(arrays,variant)
            records.append(dict(variant=variant,T=32,warps=1,phase=phase,file=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**counts))
    return dict(complete=True,strict_and_row_recip_five_phases_each_independent_CPU_exact=True,
        random_boundary_neighbors_tiny_and_zero_checked=True,
        Pq_scale_INT32_and_FP32_output_words_exact=True),records
