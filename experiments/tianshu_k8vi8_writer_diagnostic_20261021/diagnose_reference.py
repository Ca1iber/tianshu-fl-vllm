# SPDX-License-Identifier: Apache-2.0
import hashlib,json,struct
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def main():
    torch.set_num_threads(8);torch.manual_seed(202610210101)
    bits=torch.arange(65536,dtype=torch.int32).to(torch.int16);values=bits.view(torch.bfloat16)
    finite=values[torch.isfinite(values)].reshape(-1,128);random=torch.randn(8192,128,dtype=torch.bfloat16)
    random[:2048].mul_(1.e-30);random[2048:4096].mul_(1.e30)
    special=torch.zeros(2,128,dtype=torch.bfloat16);special[1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
    x=torch.cat((finite,random,special));xf=x.float();cpu_s=xf.abs().amax(-1).clamp_min(1.e-8)/127.
    true_s=torch.tensor([f32(max(max(abs(v) for v in row),f32(1.e-8))/127.) for row in xf.tolist()],dtype=torch.float32)
    cpu64_s=(xf.abs().amax(-1).clamp_min(1.e-8).double()/127.).float()
    xg=x.cuda().float();gpu_s=(xg.abs().amax(-1).clamp_min(1.e-8).double()/127.).float().cpu()
    gpu32_s=(xg.abs().amax(-1).clamp_min(1.e-8)/127.).cpu()
    scales={'CPU_FP32':cpu_s,'CPU_FP64_rounded_to_FP32':cpu64_s,'GPU_FP64_rounded_to_FP32':gpu_s,'GPU_FP32':gpu32_s}
    result=dict(complete=True,diagnostic_only=True,checked_heads=x.shape[0],scale_mismatches_vs_independent_struct_FP32={k:int((v!=true_s).sum()) for k,v in scales.items()},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    wrong=torch.where((gpu_s!=true_s)|(cpu_s!=true_s))[0][:12]
    result['samples']=[dict(row=int(i),maximum=float(xf[i].abs().max()),struct=float(true_s[i]),CPU_FP32=float(cpu_s[i]),CPU_FP64=float(cpu64_s[i]),GPU_FP64=float(gpu_s[i]),GPU_FP32=float(gpu32_s[i])) for i in wrong]
    torch.save(dict(x=x,struct_scale=true_s,**scales),Path('/workspace/benchmark-results/20261021_k8vi8_writer_diagnostic_tianshu/reference_diagnostic.pt'))
    (BASE/'REFERENCE_DIAGNOSTIC_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
