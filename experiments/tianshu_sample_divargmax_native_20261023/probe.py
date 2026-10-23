# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,json,os,signal,statistics,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve();assert not root.exists();assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True);root.mkdir(parents=True)
 remaining=int((datetime.fromisoformat('2026-10-23T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert remaining>0;signal.alarm(min(600,remaining))
 import numpy as np
 import torch
 import flag_gems
 import vllm_fl
 from vllm.v1.sample.ops.topk_topp_sampler import sample_with_exponential_noise as original
 from kernel import launch as fused
 def launch(probs,noise):
  if probs.is_contiguous() and noise.is_contiguous():return fused(probs,noise)
  scores=probs.contiguous();output=fused(scores,noise.contiguous());probs.copy_(scores);return output
 torch.set_num_threads(8)
 sources=[Path(__file__).with_name('generated_argmax.py'),Path(__file__).with_name('generate_cpu.py'),Path(__file__).with_name('SOURCE_PROOF_32g.json'),Path(__file__),Path(__file__).with_name('kernel.py'),Path(__file__).with_name('PLAN_32g.md'),Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_sampler.py'),Path('/workspace/FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/div.py'),Path('/workspace/FlagGems/src/flag_gems/ops/argmax.py')]
 snap=root/'source_snapshot';snap.mkdir();source_hash={}
 for j,f in enumerate(sources):raw=f.read_bytes();(snap/f'{j}_{f.name}').write_bytes(raw);source_hash[str(f)]=hashlib.sha256(raw).hexdigest()
 data=dict(phase='running',cases=[],timings=[],sources=source_hash,official_benchmark=False,adopted=False,model_quality_tested=False)
 def save():(root/'result.json').write_text(json.dumps(data,indent=2)+'\n')
 def bits(t):return t.detach().contiguous().view(torch.int32).cpu().numpy()
 save();torch.manual_seed(2026102355)
 try:
  with flag_gems.use_gems():
   for case_id,(b,n,strided,pattern) in enumerate([(64,130560,False,'normal'),(64,130560,True,'normal'),(8,130560,False,'normal'),(1,130560,False,'normal'),(3,130559,True,'ties'),(3,130559,True,'nan'),(3,130559,False,'infinity'),(3,13,True,'all_zero')]):
    pad=17 if strided else 0;full=torch.empty((b,n+pad),device='cuda',dtype=torch.float32);base=full[:,:n]
    base.copy_(torch.softmax(torch.randn((b,n),device='cuda'),-1));q=torch.empty_like(base);q.exponential_()
    if pattern=='ties':base.zero_();q.fill_(1);base[:,7]=base[:,-1]=1
    if pattern=='nan':base.zero_();q.fill_(1);base[:,5]=base[:,-1]=float('nan')
    if pattern=='infinity':base.fill_(1);q.fill_(1);q[:,3]=q[:,-1]=0
    if pattern=='all_zero':base.zero_();q.fill_(1)
    baseline=torch.empty_strided(base.shape,base.stride(),device='cuda',dtype=torch.float32);candidate=torch.empty_strided(base.shape,base.stride(),device='cuda',dtype=torch.float32)
    baseline.copy_(base);candidate.copy_(base);ref=original(baseline,q);out=launch(candidate,q);torch.cuda.synchronize()
    gb,gc=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
    with torch.cuda.graph(gb):baseline.copy_(base);br=original(baseline,q)
    with torch.cuda.graph(gc):candidate.copy_(base);cr=launch(candidate,q)
    saved_rng=torch.cuda.get_rng_state().clone();old=[]
    for phase in range(2):
     if phase:base.copy_(base.flip(-1));q.copy_(q.flip(-1))
     pb=bits(base);qb=bits(q);gb.replay();gc.replay();torch.cuda.synchronize();rb,cb=bits(baseline),bits(candidate);rt,ct=br.cpu().numpy(),cr.cpu().numpy()
     checks=dict(complete_score_bits_equal=bool(np.array_equal(rb,cb)),native_tokens_equal=bool(np.array_equal(rt,ct)),inputs_and_RNG_unchanged=bool(np.array_equal(pb,bits(base)) and np.array_equal(qb,bits(q)) and torch.equal(saved_rng,torch.cuda.get_rng_state())))
     fp=root/f'case{case_id}_phase{phase}.npz';np.savez(fp,probs_bits=pb,noise_bits=qb,native_score_bits=rb,candidate_score_bits=cb,native_tokens=rt,candidate_tokens=ct)
     old.append(cr.clone());data['cases'].append(dict(case=case_id,phase=phase,batch=b,vocab=n,strided=strided,pattern=pattern,**checks,file=str(fp),sha256=hashlib.sha256(fp.read_bytes()).hexdigest()));save();assert all(checks.values()),checks
    if pattern=='normal' and b==64:
     times=[[],[]]
     for repeat in range(7):
      for which in ((0,1) if repeat%2==0 else (1,0)):
       start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
       for _ in range(30):(gb if which==0 else gc).replay()
       end.record();end.synchronize();times[which].append(start.elapsed_time(end)/30)
     med=[statistics.median(t[1:]) for t in times];data['timings'].append(dict(batch=b,vocab=n,strided=strided,raw_ms=times,exclude_first_round=True,baseline_ms=med[0],candidate_ms=med[1],candidate_latency_change_percent=100*(med[1]/med[0]-1)))
    assert torch.equal(saved_rng,torch.cuda.get_rng_state());save();print(json.dumps(dict(case=case_id,phases=2,status='passed')),flush=True)
    del gb,gc,baseline,candidate,base,full,q,ref,out,br,cr,old;torch.cuda.empty_cache()
  assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in source_hash.items());data.update(phase='complete',complete=True,all_GPU_checks_passed=True)
 except BaseException as e:data.update(phase='failed',error=repr(e),traceback=traceback.format_exc());raise
 finally:save()
if __name__=='__main__':main()
