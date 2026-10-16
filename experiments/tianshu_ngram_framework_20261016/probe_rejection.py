# SPDX-License-Identifier: Apache-2.0
"""Independent CPU oracle for the installed stochastic rejection kernel."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixed',action='store_true')
    parser.add_argument('--stop-at-utc',required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    deadline=datetime.fromisoformat(args.stop_at_utc)
    remaining=int((deadline-datetime.now(timezone.utc)).total_seconds());assert remaining>0
    signal.alarm(min(300,remaining))
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    import torch
    from vllm.v1.sample import rejection_sampler as module
    source=Path(module.__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='953bb182ac2e0ef8a2c07b9ac3e157e3542f3159b3ae6f3f3f5378acfe851a86'
    import sys
    compat=Path(__file__).resolve().parents[1]/'tianshu_dspark_20261012/rejection_compat.py'
    assert hashlib.sha256(compat.read_bytes()).hexdigest()=='80df34a4e9e5fb74d46f86698d7c50d68212d7fe94d0941fc32a893fb40bf415'
    sys.path.insert(0,str(compat.parent))
    (root/'original_rejection_sampler.py').write_bytes(source.read_bytes())
    torch.manual_seed(2026101248)
    results=[]
    for counts in ([7],[3,7],[0,1,7]):
        total=sum(counts);batch=len(counts);vocab=4
        cu=torch.tensor(counts,dtype=torch.int32).cumsum(0).to('cuda')
        ids=torch.ones(total,device='cuda',dtype=torch.int32)
        q=torch.zeros(total,vocab,device='cuda');q[:,1]=1.
        p_cpu=torch.zeros(total,vocab)
        ratios=[.1,.49,.5,.51,.9,1.,0.]*batch
        p_cpu[:,1]=torch.tensor(ratios[:total])
        target=p_cpu.to('cuda');recover=torch.full((total,),2,device='cuda',dtype=torch.int32)
        bonus=torch.full((batch,),3,device='cuda',dtype=torch.int32)
        greedy=torch.zeros(batch,device='cuda',dtype=torch.bool)
        rates=torch.empty(7,device='cuda')
        values=[.05,.8,.5,.5,.95,.999,.01]*batch
        for dtype in (torch.float32,torch.float64):
            u_cpu=torch.tensor(values[:total],dtype=dtype);uniform=u_cpu.to('cuda')
            expected=torch.full((batch,8),-1,dtype=torch.int32);start=0
            for row,count in enumerate(counts):
                rejected=False
                for pos in range(count):
                    if rejected:break
                    accept=bool(float(p_cpu[start+pos,1])>=float(u_cpu[start+pos]))
                    expected[row,pos]=1 if accept else 2;rejected=not accept
                if not rejected:expected[row,count]=3
                start+=count
            actual=torch.full((batch,8),-1,device='cuda',dtype=torch.int32)
            module.rejection_random_sample_kernel[(batch,)](
                actual,cu,ids,q,target,bonus,recover,uniform,greedy,7,vocab,rates,
                NO_DRAFT_PROBS=False,SYNTHETIC_MODE=False,num_warps=4)
            observed=actual.cpu()
            results.append({'counts':counts,'uniform_dtype':str(dtype),
                'uniform_host_roundtrip_equal':torch.equal(uniform.cpu(),u_cpu),
                'cpu_expected':expected.tolist(),'gpu_observed':observed.tolist(),
                'bitwise_cpu_equal':torch.equal(observed,expected)})
    uniform=module.generate_uniform_probs(512,[512],{},torch.device('cuda')).cpu()
    result={'complete':True,'official_benchmark':False,'accuracy_test':False,
            'installed_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'cases':results,'all_passed':all(x['bitwise_cpu_equal'] for x in results),
            'fp64_uniform_host_min':uniform.min().item(),'fp64_uniform_host_max':uniform.max().item(),
            'fp64_uniform_unique':uniform.unique().numel(),
            'note':'Only acceptance threshold and original kernel semantics are checked; no full sampling quality claim.'}
    if args.fixed:
        import os
        os.environ['TIANSHU_DSPARK_RECORD_DIR']=str(root/'adapter_records')
        from rejection_compat import install
        install(module);fixed=[]
        for counts in ([7],[3,7],[0,1,7],[7]*64):
            total=sum(counts);batch=len(counts);vocab=4
            cu=torch.tensor(counts,dtype=torch.int32).cumsum(0).to('cuda')
            ids=torch.ones(total,device='cuda',dtype=torch.int32)
            bonus=torch.full((batch,),3,device='cuda',dtype=torch.int32)
            recover=torch.full((total,),2,device='cuda',dtype=torch.int32)
            rates=torch.empty(7,device='cuda')
            for no_draft in (False,True):
                for mixed_greedy in (False,True):
                    for seed in (17,73,139):
                        torch.manual_seed(seed)
                        generators={1:torch.Generator(device='cuda').manual_seed(seed+1)} if batch>1 else {}
                        u=module.generate_uniform_probs(total,counts,generators,torch.device('cuda'))
                        host=u.cpu();assert u.dtype==torch.float32 and bool((host>0).all() and (host<=1).all())
                        p_cpu=torch.zeros(total,vocab);q_cpu=torch.zeros(total,vocab)
                        values=torch.tensor([0.,.1,.49,.5,.51,.9,1.])
                        p_cpu[:,1]=values[torch.arange(total)%7]
                        q_cpu[:,1]=.5;q_cpu[torch.arange(total)%11==0,1]=0.
                        ratios=p_cpu[:,1] if no_draft else p_cpu[:,1]/q_cpu[:,1]
                        greedy_cpu=torch.arange(batch)%2==1 if mixed_greedy else torch.zeros(batch,dtype=torch.bool)
                        expected=torch.full((batch,8),-1,dtype=torch.int32);start=0
                        for row,count in enumerate(counts):
                            if greedy_cpu[row]:start+=count;continue
                            rejected=False
                            for pos in range(count):
                                if rejected:break
                                idx=start+pos
                                accept=bool((no_draft or q_cpu[idx,1]>0) and ratios[idx]>=host[idx])
                                expected[row,pos]=1 if accept else 2;rejected=not accept
                            if not rejected:expected[row,count]=3
                            start+=count
                        actual=torch.full((batch,8),-1,device='cuda',dtype=torch.int32)
                        module.rejection_random_sample_kernel[(batch,)](
                            actual,cu,ids,q_cpu.to('cuda'),p_cpu.to('cuda'),bonus,recover,u,
                            greedy_cpu.to('cuda'),7,vocab,rates,NO_DRAFT_PROBS=no_draft,
                            SYNTHETIC_MODE=False,num_warps=4)
                        passed=torch.equal(actual.cpu(),expected)
                        fixed.append({'counts':counts,'seed':seed,'no_draft_probs':no_draft,
                            'mixed_greedy':mixed_greedy,'bitwise_cpu_equal':passed,
                            'min_threshold':host.min().item(),'max_threshold':host.max().item()})
                        assert passed,fixed[-1]
        generated=module.generate_uniform_probs(65536,[65536],{},torch.device('cuda')).cpu()
        result['fixed_cases']=fixed;result['fixed_all_passed']=all(x['bitwise_cpu_equal'] for x in fixed)
        result['fixed_uniform_summary']={'minimum':generated.min().item(),'maximum':generated.max().item(),
            'unique':generated.unique().numel(),'mean':generated.mean().item(),
            'below_half_fraction':(generated<.5).float().mean().item()}
        assert .49<generated.mean().item()<.51 and generated.unique().numel()>60000
    result['source_snapshots']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(),source,compat)}
    for name in result['source_snapshots']:
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
