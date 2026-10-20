# SPDX-License-Identifier: Apache-2.0
"""Preserve failed component; independently check CPU versus GPU reference division."""
import hashlib,json,struct,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_k8vi8_heads_20261021'
sys.path.insert(0,str(PARENT))
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
from layout import allocate
import writer

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encode(x):
    bits=x.view(torch.int16).to(torch.int32);mag=bits&32767;rounded=(mag+7+((mag>>4)&1))>>4
    return (torch.where(rounded>=936,(rounded-928).clamp(max=127),0)|((bits&32768)>>8)).to(torch.uint8)
def pack(x):
    xf=x.float();scale=xf.abs().amax(-1).clamp_min(1.e-8)/127.
    return (xf/scale[...,None]).round().clamp(-127,127).to(torch.int8),scale

def main():
    root=Path('/workspace/benchmark-results/20261021_k8vi8_writer_diagnostic_tianshu');assert not root.exists();root.mkdir()
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
    sources=json.loads((PARENT/'CPU_GATE_32g.json').read_text())['sources'];sources[str(Path(__file__).resolve())]=sha(__file__)
    assert all(sha(p)==v for p,v in sources.items())
    torch.manual_seed(202610210100)
    key=torch.randn(11,2,128,device='cuda',dtype=torch.bfloat16);value=torch.randn_like(key)
    value[0,0].zero_();value[0,1].zero_();value[0,1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],device='cuda',dtype=torch.bfloat16)
    value[2].mul_(1.e-30);value[3].mul_(1.e30);value[4].zero_()
    slots=torch.tensor([0,1,18,31,32,47,48,63,-1,-33,-999],device='cuda',dtype=torch.int64)
    raw,native,k,v,ks,vs,layout=allocate(torch,FullAttentionSpec,_reshape_attention_kv_cache,'K8VI8Heads',5,False)
    before_cpu=torch.get_rng_state().clone();before_gpu=torch.cuda.get_rng_state().clone()
    compiled=writer.write(key,value,k,v,slots,ks,vs);torch.cuda.synchronize()
    assert torch.equal(torch.get_rng_state(),before_cpu) and torch.equal(torch.cuda.get_rng_state(),before_gpu)
    blocks,tokens=slots[:8]//16,slots[:8]%16;bc,tc=blocks.cpu(),tokens.cpu()
    actual_v=v[blocks,tokens].cpu();actual_vs=vs[blocks,tokens].cpu();actual_ks=ks[blocks,tokens].cpu();actual_k=k[blocks,tokens].cpu()
    cpu_q,cpu_s=pack(value[:8].cpu());gpu_q,gpu_s=pack(value[:8]);gpu_q,gpu_s=gpu_q.cpu(),gpu_s.cpu()
    reference_raw,_,rk,rv,rks,rvs,_=allocate(torch,FullAttentionSpec,_reshape_attention_kv_cache,'K8VI8Heads',5,False,device='cpu')
    rk[bc,tc]=encode(key[:8].cpu());rv[bc,tc]=cpu_q;rks[bc,tc]=1.;rvs[bc,tc]=cpu_s
    raw_actual=raw.cpu();cpu_mismatch=int((raw_actual!=reference_raw).sum())
    rv[bc,tc]=gpu_q;rvs[bc,tc]=gpu_s;gpu_mismatch=int((raw_actual!=reference_raw).sum())
    def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
    exact_zero_scale=f32(f32(1.e-8)/127.)
    record=dict(complete=True,diagnostic_only=True,actual_full_physical_bytes_match_CPU_RNE_reference=cpu_mismatch==0,
        CPU_reference_byte_mismatches=cpu_mismatch,GPU_reference_byte_mismatches=gpu_mismatch,
        actual_V_integer_vs_CPU_mismatches=int((actual_v!=cpu_q).sum()),actual_V_integer_vs_GPU_mismatches=int((actual_v!=gpu_q).sum()),
        actual_scale_vs_CPU_mismatches=int((actual_vs.view(torch.int32)!=cpu_s.view(torch.int32)).sum()),
        actual_scale_vs_GPU_mismatches=int((actual_vs.view(torch.int32)!=gpu_s.view(torch.int32)).sum()),
        actual_K_bytes_exact=bool(torch.equal(actual_k,encode(key[:8].cpu()))),actual_K_scales_one=bool((actual_ks==1.).all()),
        zero_scale_actual=float(actual_vs[0,0]),zero_scale_CPU=float(cpu_s[0,0]),zero_scale_GPU=float(gpu_s[0,0]),zero_scale_independent_struct_FP32=exact_zero_scale,
        all_sources=sources,layout=layout,RNG_unchanged=True,original_failed_component_not_modified=True)
    torch.save(dict(key=key.cpu(),value=value.cpu(),slots=slots.cpu(),actual_raw=raw_actual,actual_V=actual_v,actual_V_scale=actual_vs,
        CPU_V=cpu_q,CPU_scale=cpu_s,GPU_V=gpu_q,GPU_scale=gpu_s),root/'diagnostic.pt')
    record['raw_fixture_sha256']=sha(root/'diagnostic.pt');record['raw_fixture_bytes']=(root/'diagnostic.pt').stat().st_size
    assert all(sha(p)==v for p,v in sources.items())
    (root/'result.json').write_text(json.dumps(record,indent=2)+'\n');(BASE/'DIAGNOSTIC_GATE_32g.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('all_sources','layout')}))
if __name__=='__main__':main()
