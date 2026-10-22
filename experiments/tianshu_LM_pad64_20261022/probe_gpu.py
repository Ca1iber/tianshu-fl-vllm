# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,importlib.util,inspect,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args();remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>900;root=args.output;assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware);freeze=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());sources=freeze['sources'];assert all(sha(p)==h for p,h in sources.items())
 for p,h in sources.items():
  d=root/'source_snapshot'/Path(p).relative_to('/');d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(Path(p).read_bytes());assert sha(d)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,phases=[],rounds=[],actual_vendor_traces={},checkpoint_before=sha(MODEL),model_executed=False,adopted=False)
 def save():
  t=root/'result.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(root/'result.json')
 def stop(signum,frame):raise TimeoutError(str(signum))
 for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 signal.alarm(min(remain,900));save()
 try:
  import torch,vllm_fl,flag_gems
  from safetensors import safe_open
  from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
  from vllm_fl.utils import get_flag_gems_whitelist_blacklist
  torch.set_num_threads(4);white,black=get_flag_gems_whitelist_blacklist();assert not white and not black;assert str(Path(inspect.getsourcefile(flag_gems.linear)).resolve())==freeze['actual_vendor_linear_source'];flag_gems.enable(record=False,once=True);method=UnquantizedEmbeddingMethod();old=load('_pad64_old',EXP/'tianshu_dense_w8a8_20261020/kernel.py');new=load('_pad64_new',BASE/'kernel.py');n,k=130560,2048;cfg=dict(bm=64,bn=128,bk=128,warps=4,stages=1)
  with safe_open(str(MODEL),framework='pt',device='cpu') as cp:weight=cp.get_tensor('lm_head.weight').contiguous()
  wf=weight.float();sc=wf.abs().amax(1).clamp_min(1.e-8)/127.;codes=(wf/sc[:,None]).round().clamp(-127,127).to(torch.int8);del wf
  def th(t):return ah(t.contiguous().view(torch.uint8).cpu().numpy())
  proof=next(r for r in json.loads((EXP/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text()) if r['prefix']=='lm_head');state['weight_signatures']={key:th(t) for key,t in [('original_BF16_sha256',weight),('packed_sha256',codes),('scales_sha256',sc)]};assert all(proof[key]==h for key,h in state['weight_signatures'].items());w=codes.cuda();ws=sc.cuda();bfweight=weight.cuda();layer=SimpleNamespace(weight=bfweight,prefix='lm_head');dummy=torch.empty(1,device='cuda',dtype=torch.int32);rows_by_m={m:np.unique(np.linspace(0,m-1,8).astype(np.int64)) for m in freeze['M_points']};cols=np.unique(np.concatenate([np.linspace(0,n-1,64).astype(np.int64),np.array([0,1,1023,1024,65535,65536,32767,32768,n-2,n-1])]))
  assert len(cols)==72
  for m in freeze['M_points']:
   guards=[]
   def alloc(shape,dtype,value):
    length=int(np.prod(shape));b=torch.full((length+512,),value,dtype=dtype,device='cuda');guards.append((b,value));return b[256:256+length].view(shape)
   a=alloc((m,k),torch.bfloat16,17.);padded=alloc((64,k),torch.bfloat16,17.);q=alloc((64,k),torch.int8,17);s=alloc((64,),torch.float32,17.);rq=alloc((64,k),torch.int8,17);rs=alloc((64,),torch.float32,17.);integer=alloc((64,n),torch.int32,-777);refint=alloc((64,n),torch.int32,-777);bfout=alloc((64,n),torch.bfloat16,17.);fpout=alloc((64,n),torch.float32,17.)
   def fixture(phase,restore=False):
    seed=202610242100+100*m+phase;sample=np.random.default_rng(seed).standard_normal((m,k),dtype=np.float32);b=sample.view(np.uint32);words=((b+32767+((b>>16)&1))>>16).astype(np.uint16);words[0]=0;a.copy_(torch.from_numpy(words).view(torch.bfloat16));padded.zero_();padded[:m].copy_(a);sign=1 if restore or phase!=1 else -1;factor=1. if restore or phase!=2 else .5;w.copy_(codes*sign);ws.copy_(sc*factor);bfweight.copy_(weight*(sign*factor));return seed,sign,factor
   def control():
    if m==64:old.quantize(a,rq,rs);old.mm(rq,w,rs,ws,bfout,dummy,cfg);return bfout.float()
    return method.apply(layer,a,None).float()
   def candidate():new.quantize_padded(a,q,s);new.mm(q,w,s,ws,fpout,dummy,cfg);return fpout[:m]
   fixture(0);traces=[]
   def trace(frame,event,arg):
    if event=='call' and 'flag_gems' in frame.f_code.co_filename and frame.f_code.co_name in ('linear','mm','_launch_mm'):traces.append(dict(function=frame.f_code.co_name,source=frame.f_code.co_filename,line=frame.f_code.co_firstlineno))
   sys.setprofile(trace)
   try:control()
   finally:sys.setprofile(None)
   if m<64:assert any(Path(r['source']).resolve()==Path(freeze['actual_vendor_linear_source']).resolve() for r in traces)
   state['actual_vendor_traces'][str(m)]=traces;graphs={};graph_outputs={};ops={'actual_control':control,'padded_fused':candidate}
   qkernel=new.quantize_padded(a,q,s);gkernel=new.mm(q,w,s,ws,fpout,dummy,cfg)
   state.setdefault("compiled",{})[str(m)]={name:dict(binary_sha256=hashlib.sha256(ker.kernel).hexdigest(),cache_hash=ker.hash,n_regs=ker.n_regs,n_spills=ker.n_spills,shared=ker.metadata.shared) for name,ker in (("quantizer",qkernel),("GEMM",gkernel))}
   for name,fn in ops.items():
    fn();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    graphs[name]=graph;graph_outputs[name]=out
   for phase in range(3):
    seed,sign,factor=fixture(phase);immutable=[t.clone() for t in (a,w,ws,bfweight)];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();old.quantize(padded,rq,rs);old.mm(rq,w,rs,ws,bfout,refint,cfg,return_int=True);reference=bfout.float().clone();new.quantize_padded(a,q,s);assert torch.equal(q,rq) and torch.equal(s,rs);new.mm(q,w,s,ws,fpout,integer,cfg,return_int=True);assert torch.equal(integer,refint) and torch.equal(fpout,reference) and not bool(fpout[m:].count_nonzero())
    baseline=control();torch.cuda.synchronize();base_copy=baseline.clone()
    if m==64:assert torch.equal(base_copy,reference)
    candidate();assert torch.equal(fpout,reference)
    for name in ops:
     graph_outputs[name].fill_(float('nan'));q.fill_(117);s.fill_(float('nan'));fpout.fill_(float('nan'));graphs[name].replay();torch.cuda.synchronize();assert torch.equal(graph_outputs[name],base_copy if name=='actual_control' else reference[:m])
    assert torch.equal(fpout,reference)
    for t,before in zip((a,w,ws,bfweight),immutable):assert torch.equal(t,before)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    for b,value in guards:assert bool((b[:256]==value).all() and (b[-256:]==value).all())
    rows=rows_by_m[m];rg=torch.tensor(rows,device='cuda');cg=torch.tensor(cols,device='cuda');art=root/f'M{m}_phase{phase}.npz';np.savez(art,rows=rows,columns=cols,integer_selected=refint[rg[:,None],cg[None,:]].cpu().numpy(),output_FP32_selected=reference[rg[:,None],cg[None,:]].cpu().numpy(),baseline_FP32_selected=base_copy[rg[:,None],cg[None,:]].cpu().numpy(),Aq=q.cpu().numpy(),As=s.cpu().numpy());state['phases'].append(dict(M=m,phase=phase,seed=seed,sign=sign,factor=factor,artifact=art.name,sha256=sha(art),all64_q_and_scale_exact=True,full_INT32_exact=True,full_FP32_oracle_exact=True,original_M64_bit_exact=(m==64),full_values=64*n,graph_actual_return_poison=True,all_inputs_weights_RNG_guards=True));save()
   fixture(0,restore=True)
   for mode in ('eager','graph'):
    for roundid in range(7):
     order=list(ops) if roundid%2==0 else list(reversed(ops));times={}
     for name in order:
      fn=graphs[name].replay if mode=='graph' else ops[name];start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
      for _ in range(20):fn()
      end.record();end.synchronize();times[name]=start.elapsed_time(end)/20
     state['rounds'].append(dict(M=m,mode=mode,round=roundid,excluded=roundid==0,milliseconds=times,order=order));save()
   assert th(w)==proof['packed_sha256'] and th(ws)==proof['scales_sha256'] and th(bfweight)==proof['original_BF16_sha256'];print(json.dumps(dict(M=m,complete=True)),flush=True)
  assert all(sha(p)==h for p,h in sources.items()) and sha(MODEL)==state['checkpoint_before'];state.update(complete=True,phase='complete',checkpoint_after=sha(MODEL));save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
