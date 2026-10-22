# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args();remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>900;signal.alarm(min(remain,900));root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
 freeze=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());sources=freeze['sources'];assert freeze['complete'] and all(sha(p)==h for p,h in sources.items())
 for path,h in sources.items():
  target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,phases=[],rounds=[],model_executed=False,official_benchmark=False,adopted=False,checkpoint_sha256_before=sha(MODEL))
 def save():
  t=root/'result.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(root/'result.json')
 save()
 def deadline(signum,frame):raise TimeoutError(f"component stop signal {signum}")
 for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,deadline)
 try:
  import torch,vllm_fl
  from safetensors import safe_open
  torch.set_num_threads(4);old=load('_lmcast_original',EXP/'tianshu_dense_w8a8_20261020/kernel.py');new=load('_lmcast_fused',BASE/'kernel.py');m,n,k=64,130560,2048;config=freeze['actual_r3_LM64']['configuration']
  with safe_open(str(MODEL),framework='pt',device='cpu') as cp:weight=cp.get_tensor('lm_head.weight').contiguous()
  wf=weight.float();sc=wf.abs().amax(1).clamp_min(1.e-8)/127.;codes=(wf/sc[:,None]).round().clamp(-127,127).to(torch.int8);del wf
  def th(t):return ah(t.contiguous().view(torch.uint8).cpu().numpy())
  expected=next(r for r in json.loads((EXP/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text()) if r['prefix']=='lm_head');state['weight_signatures']={name:th(t) for name,t in [('original_BF16_sha256',weight),('packed_sha256',codes),('scales_sha256',sc)]};assert all(expected[key]==h for key,h in state['weight_signatures'].items())
  guards=[]
  def alloc(shape,dtype,value):
   size=int(np.prod(shape));buf=torch.full((size+512,),value,dtype=dtype,device='cuda');guards.append((buf,value));return buf[256:size+256].view(shape)
  a=alloc((m,k),torch.bfloat16,17.);q=alloc((m,k),torch.int8,17);asc=alloc((m,),torch.float32,17.);w=alloc((n,k),torch.int8,17);ws=alloc((n,),torch.float32,17.);integer=alloc((m,n),torch.int32,-777);refint=alloc((m,n),torch.int32,-777);bf16=alloc((m,n),torch.bfloat16,17.);fp32={name:alloc((m,n),torch.float32,17.) for name in ('fused_Triton','native_B512','native_B1024')};dummy=torch.empty(1,device='cuda',dtype=torch.int32)
  w.copy_(codes);ws.copy_(sc);a.zero_()
  def control():old.quantize(a,q,asc);old.mm(q,w,asc,ws,bf16,dummy,config);return bf16.to(torch.float32)
  def fused():old.quantize(a,q,asc);new.mm(q,w,asc,ws,fp32['fused_Triton'],dummy,config);return fp32['fused_Triton']
  def native(name,block):old.quantize(a,q,asc);torch._int_mm(q,w.T,out=integer);new.native_dequant(integer,asc,ws,fp32[name],block);return fp32[name]
  operations={'control':control,'fused_Triton':fused,'native_B512':lambda:native('native_B512',512),'native_B1024':lambda:native('native_B1024',1024)};graphs={};graph_outputs={}
  actual=old.mm(q,w,asc,ws,bf16,dummy,config);state['control_compiled']=dict(source_sha256=sha(EXP/'tianshu_dense_w8a8_20261020/kernel.py'),binary_sha256=hashlib.sha256(actual.kernel).hexdigest(),n_regs=actual.n_regs,n_spills=actual.n_spills);assert state['control_compiled']['source_sha256']==freeze['actual_r3_LM64']['kernel_sha256'] and actual.n_regs==93 and actual.n_spills==0
  for name,fn in operations.items():
   fn();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=fn()
   graphs[name]=graph;graph_outputs[name]=out
  rows=np.unique(np.linspace(0,m-1,8).astype(np.int64));columns=np.unique(np.concatenate([np.linspace(0,n-1,64).astype(np.int64),np.array([0,1,1023,1024,65535,65536,32767,32768,n-2,n-1])]))
  state.update(phase='numerical',M=m,N=n,K=k,config=config,candidate_writes_FP32_after_explicit_BF16_round=True);save()
  for phase in range(3):
   seed=202610240100+phase;sample=np.random.default_rng(seed).standard_normal((m,k),dtype=np.float32);bits=sample.view(np.uint32);words=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);words[0]=0;ac=torch.from_numpy(words.copy()).view(torch.bfloat16);sign=-1 if phase==1 else 1;factor=.5 if phase==2 else 1.;a.copy_(ac);w.copy_(codes*sign);ws.copy_(sc*factor);immutable=[t.clone() for t in (a,w,ws)];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
   old.quantize(a,q,asc);old.mm(q,w,asc,ws,bf16,refint,config,return_int=True);reference=bf16.float().clone();new.mm(q,w,asc,ws,fp32['fused_Triton'],integer,config,return_int=True);assert torch.equal(integer,refint) and torch.equal(fp32['fused_Triton'],reference);entries=[]
   for name,fn in operations.items():
    eager=fn();torch.cuda.synchronize();assert eager.dtype==torch.float32 and eager.shape==(m,n) and torch.equal(eager,reference)
    graph_outputs[name].fill_(float('nan'));bf16.fill_(float('nan'));integer.fill_(-2147483647);q.fill_(117);asc.fill_(float('nan'));graphs[name].replay();torch.cuda.synchronize();assert torch.equal(graph_outputs[name],reference)
    if name.startswith('native'):assert torch.equal(integer,refint)
    entries.append(dict(name=name,full_FP32_bit_exact=True,full_values=m*n,graph_actual_return_poison_matches_eager=True,eager_data_ptr=eager.data_ptr(),graph_data_ptr=graph_outputs[name].data_ptr(),full_native_INT32_exact=name.startswith('native')))
   for t,before in zip((a,w,ws),immutable):assert torch.equal(t,before)
   assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
   for buf,value in guards:assert bool((buf[:256]==value).all() and (buf[-256:]==value).all())
   artifact=root/f'phase{phase}.npz';np.savez(artifact,rows=rows,columns=columns,integer_selected=refint[torch.tensor(rows,device='cuda')[:,None],torch.tensor(columns,device='cuda')[None,:]].cpu().numpy(),output_FP32_selected=reference[torch.tensor(rows,device='cuda')[:,None],torch.tensor(columns,device='cuda')[None,:]].cpu().numpy(),Aq=q.cpu().numpy(),As=asc.cpu().numpy());state['phases'].append(dict(phase=phase,seed=seed,weight_sign=sign,weight_scale_factor=factor,artifact=artifact.name,sha256=sha(artifact),entries=entries,inputs_weights_RNG_guards_unchanged=True));save()
  state['phase']='timing';save()
  for mode in ('graph','eager'):
   for roundid in range(7):
    names=list(operations);shift=roundid%len(names);names=names[shift:]+names[:shift]
    if roundid%2:names.reverse()
    values={}
    for name in names:
     fn=graphs[name].replay if mode=='graph' else operations[name];start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
     for _ in range(20):fn()
     end.record();end.synchronize();values[name]=start.elapsed_time(end)/20
    state['rounds'].append(dict(mode=mode,round=roundid,excluded=roundid==0,order=names,repeats=20,milliseconds=values));save()
  for tensor,before in zip((a,w,ws),immutable):assert torch.equal(tensor,before)
  for buf,value in guards:assert bool((buf[:256]==value).all() and (buf[-256:]==value).all())
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
  assert all(sha(p)==h for p,h in sources.items());state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];state.update(complete=True,phase='complete');save();print(json.dumps(dict(complete=True,phases=3,variants=3)),flush=True)
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
if __name__=='__main__':main()
