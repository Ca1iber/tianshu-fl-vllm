# SPDX-License-Identifier: Apache-2.0
"""CPU-reference checks for the DSpark draft's RMSNorm and query-only RoPE."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output;assert not root.exists();root.mkdir(parents=True)
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    import torch
    import vllm_fl
    from tianshu_dspark_adapter.model import DSparkOpsProxy
    import vllm._custom_ops as original
    ops=DSparkOpsProxy(original)
    rows=[]

    def check(a,b):
        a,b=a.cpu().float(),b.cpu().float();diff=(a-b).abs();bad=diff>.02+.01*b.abs()
        return {'passed':bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),
                'failed_elements':bad.sum().item(),'max_abs':diff.max().item(),'atol':.02,'rtol':.01}

    torch.manual_seed(2026101207)
    for n in (1,17,2048):
        for shape in ((n,2048),(n,2,128)):
            x=torch.randn(shape,dtype=torch.bfloat16,device='cuda')
            weight=torch.randn(shape[-1],dtype=x.dtype,device=x.device)
            out=torch.empty_like(x);ops.rms_norm(out,x,weight,1e-6)
            cpu=x.cpu().float();w=weight.cpu().float()
            expected=(cpu*torch.rsqrt(cpu.square().mean(-1,keepdim=True)+1e-6)*w).to(x.dtype)
            rows.append({'operation':'rms_norm','shape':shape,'cpu_reference':check(out,expected)})
        angles=torch.randn(64,64,dtype=torch.float32)
        cache=torch.cat([angles.cos(),angles.sin()],dim=-1).to(torch.bfloat16).cuda()
        positions=torch.arange(n,device='cuda')%64
        for heads in (2,16):
            for neox in (False,True):
                size=n*heads*128
                storage=torch.full((size+512,),17.,dtype=torch.bfloat16,device='cuda')
                query=storage[:size].view(n,heads*128);query.copy_(torch.randn_like(query))
                initial=query.cpu().float().view(n,heads,128)
                c,s=cache.cpu().float()[positions.cpu()].chunk(2,-1)
                if neox:
                    rotated=torch.cat([-initial[...,64:],initial[...,:64]],dim=-1)
                    c,s=c.repeat(1,2),s.repeat(1,2)
                else:
                    rotated=torch.stack([-initial[...,1::2],initial[...,::2]],dim=-1).flatten(-2)
                    c,s=c.repeat_interleave(2,-1),s.repeat_interleave(2,-1)
                expected=(initial*c[:,None]+rotated*s[:,None]).to(query.dtype).view(n,heads*128)
                ops.rotary_embedding(positions,query,None,128,cache,neox)
                rows.append({'operation':'query_only_rope','tokens':n,'heads':heads,'neox':neox,
                             'cpu_reference':check(query,expected),
                             'output_tail_preserved':bool((storage[size:]==17.).all())})
    data={'complete':True,'cases':rows,'official_benchmark':False,
          'passed':all(r['cpu_reference']['passed'] and r.get('output_tail_preserved',True) for r in rows)}
    (root/'result.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
    assert data['passed']


if __name__=='__main__':main()
