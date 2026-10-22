# SPDX-License-Identifier: Apache-2.0
import hashlib,json,os,signal,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261022_native_weight_only_contract_GPU_tianshu')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def f32(w):return (np.asarray(w,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16(v):
 b=np.asarray(v,np.float32).view(np.uint32);return ((b+32767+((b>>16)&1))>>16).astype(np.uint16)

def main():
 remain=int((datetime.fromisoformat('2026-10-22T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert remain>600
 def stopped(signum,frame):raise TimeoutError(str(signum))
 signal.signal(signal.SIGALRM,stopped);signal.signal(signal.SIGTERM,stopped);signal.alarm(min(remain,300));assert not ROOT.exists();ROOT.mkdir();hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(ROOT/'hardware_before.txt').write_text(hardware);state=dict(complete=False,phase='initializing',model_read=False,performance_tested=False,adopted=False,cases=[])
 def save():
  temp=ROOT/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(ROOT/'result.json')
 save()
 try:
  import torch
  torch.set_num_threads(4);provider=Path(torch.__file__).resolve().parent;paths=[BASE/'PLAN_32g.md',BASE/'probe_gpu.py',provider/'_meta_registrations.py',provider/'lib/libtorch_cuda.so',provider/'lib/libtorch_cpu.so'];sources={str(p):sha(p) for p in paths};state.update(sources=sources,torch_version=torch.__version__,target='BI-V150/CoreX71');save()
  for p,h in sources.items():
   if Path(p).is_relative_to(BASE):target=ROOT/'source_snapshot'/Path(p).name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  for m in (8,64,128):
   for layout in ('NT_contiguous','NN_transposed_view'):
    for scales_dtype in ('FP32','BF16'):
     case=dict(M=m,N=256,K=256,layout=layout,scales_dtype=scales_dtype,accepted=False,phases=[]);state['cases'].append(case);seed=202610236000+m;outputs=[];x=torch.empty((m,256),device='cuda',dtype=torch.bfloat16);backing=torch.empty((256,256),device='cuda',dtype=torch.int8);w=backing if layout=='NT_contiguous' else backing.T;scales=torch.empty(256,device='cuda',dtype=torch.float32 if scales_dtype=='FP32' else torch.bfloat16)
     def fixture(phase):
      rng=np.random.default_rng(seed+phase);xb=bf16(rng.standard_normal((m,256),dtype=np.float32));xb[0]=0;wc=rng.integers(-127,128,(256,256),dtype=np.int16).astype(np.int8);sv=(np.arange(256,dtype=np.float32)%31+1)/(np.float32(257.) if phase==0 else np.float32(263.));sv=f32(bf16(sv)) if scales_dtype=='BF16' else sv;x.copy_(torch.from_numpy(xb).view(torch.bfloat16));w.copy_(torch.from_numpy(wc));scales.copy_(torch.from_numpy(sv));return xb,wc,sv
     fixture(0)
     try:torch.ops.aten._weight_int8pack_mm.default(x,w,scales);torch.cuda.synchronize()
     except RuntimeError as error:
      message=str(error);case.update(error=message,host_or_hardware_contract_rejected=True);save()
      if any(word in message.lower() for word in ('illegal memory','device-side assert','launch failed','unspecified launch')):raise
      continue
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):captured=torch.ops.aten._weight_int8pack_mm.default(x,w,scales)
     case['accepted']=True
     for phase in range(2):
      xb,wc,sv=fixture(phase);before=[t.clone() for t in (x,w,scales)];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();out=torch.ops.aten._weight_int8pack_mm.default(x,w,scales);torch.cuda.synchronize();expected=words(out);captured.fill_(float('nan'));graph.replay();torch.cuda.synchronize();assert np.array_equal(words(captured),expected) and bool(torch.isfinite(captured).all()) and bool((captured[0]==0).all());assert all(torch.equal(a,b) for a,b in zip((x,w,scales),before)) and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());h=f32(xb).astype(np.float64);codes=wc.astype(np.float64);s=sv.astype(np.float64);dot=h@codes.T;preBF16=f32(bf16(codes*s[:,None])).astype(np.float64);preFP16=np.asarray(codes*s[:,None],np.float16).astype(np.float64);refs=dict(post_scale=dot*s[None,:],pre_weight_BF16=h@preBF16.T,pre_weight_FP16=h@preFP16.T,post_dot_BF16=f32(bf16(dot)).astype(np.float64)*s[None,:]);absolute=abs(h)@abs(codes).T*s[None,:];eps=2.**-24;gamma=(2*256+4)*eps/(1-(2*256+4)*eps);exp=((expected&0x7f80)>>7).astype(np.int32);ulp=np.exp2(np.where(exp==0,-133,exp-134).astype(np.float64));bound=gamma*absolute+.5*ulp+1e-10;actual=f32(expected).astype(np.float64);classifications={name:dict(all_cells_within_predeclared_bound=bool(np.all(abs(actual-reference)<=bound)),relative_L2=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))) for name,reference in refs.items()};assert any(v['all_cells_within_predeclared_bound'] for v in classifications.values());path=ROOT/f'M{m}_{layout}_{scales_dtype}_phase{phase}.npz';np.savez(path,X_words=xb,W_codes=wc,scales=sv,O_words=expected);case['phases'].append(dict(phase=phase,artifact=path.name,sha256=sha(path),output_sha256=ahash(expected),full_CPU_FP64_classifications=classifications,graph_eager_and_guards_passed=True));save()
     save()
  assert all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete',accepted_cases=sum(c['accepted'] for c in state['cases']),contract_rejections=sum(not c['accepted'] for c in state['cases']),native_provider_and_source_hashes_unchanged=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:signal.alarm(0);(ROOT/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
