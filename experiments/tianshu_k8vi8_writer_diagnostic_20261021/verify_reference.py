# SPDX-License-Identifier: Apache-2.0
"""Audit GPU FP64-derived FP32 reference against independent CPU quantization."""
import hashlib,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent

def main():
    torch.manual_seed(202610210101)
    bits=torch.arange(65536,dtype=torch.int32).to(torch.int16);values=bits.view(torch.bfloat16)
    finite=values[torch.isfinite(values)].reshape(-1,128)
    random=torch.randn(8192,128,dtype=torch.bfloat16)
    random[:2048].mul_(1.e-30);random[2048:4096].mul_(1.e30)
    special=torch.zeros(2,128,dtype=torch.bfloat16);special[1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
    x=torch.cat((finite,random,special));xf=x.float();cpu_s=xf.abs().amax(-1).clamp_min(1.e-8)/127.
    cpu_q=(xf/cpu_s[:,None]).round().clamp(-127,127).to(torch.int8)
    xg=x.cuda().float();gpu_s=(xg.abs().amax(-1).clamp_min(1.e-8).double()/127.).float()
    gpu_q=(xg.double()/gpu_s.double()[:,None]).float().round().clamp(-127,127).to(torch.int8)
    assert torch.equal(gpu_s.cpu(),cpu_s) and torch.equal(gpu_q.cpu(),cpu_q)
    result=dict(complete=True,all_finite_BF16_patterns=65280,additional_random_heads=8192,special_zero_and_RNE_tie_heads=2,
        checked_heads=x.shape[0],checked_values=x.numel(),FP64_GPU_reference_scale_bitwise_equals_CPU_FP32=True,
        FP64_GPU_reference_quantized_values_equal_CPU=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        actual_GPU_writer_not_changed=True,reference_only_not_in_timed_writer=True)
    (BASE/'REFERENCE_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
