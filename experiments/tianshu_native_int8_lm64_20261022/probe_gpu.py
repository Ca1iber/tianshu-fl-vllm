# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--stop-at-utc',required=True);args=p.parse_args();remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>1800;signal.alarm(min(remain,1800))
 root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(root/'hardware_before.txt').write_text(hw)
 freeze=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());sources=freeze['sources'];assert freeze['complete'] and all(sha(p)==h for p,h in sources.items())
 for path,h in sources.items():
  target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,phases=[],rounds=[],breakdowns=[],model_executed=False,official_benchmark=False,adopted=False,checkpoint_sha256_before=sha(MODEL))
 def save():
  temp=root/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'result.json')
 save()
 try:
  import torch,vllm_fl,triton
  from safetensors import safe_open
  torch.set_num_threads(4);original=load('_lm64_current',ROOT/'tianshu_dense_w8a8_20261020/kernel.py');new=load('_lm64_native',BASE/'kernel.py');m,n,k=64,130560,2048
  selected=next(row for row in json.loads((ROOT/'tianshu_dense_w8a8_20261020/MODEL_INTEGRATION_PLAN_32g.json').read_text())['selected_configurations'] if row['case']=='lm_head' and row['M']==64);config=selected['configuration'];assert config==dict(bm=64,bn=128,bk=128,warps=4,stages=1)
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint:weight=checkpoint.get_tensor('lm_head.weight').contiguous()
  assert weight.shape==(n,k);wf=weight.float();scales=wf.abs().amax(1).clamp_min(1.e-8)/127.;codes=(wf/scales[:,None]).round().clamp(-127,127).to(torch.int8)
  def thash(t):return ahash(t.contiguous().view(torch.uint8).numpy())
  state['weight_signature']={key:thash(t) for key,t in [('original_BF16_sha256',weight),('packed_sha256',codes),('scales_sha256',scales)]};expected=next(row for row in json.loads((ROOT/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text()) if row['prefix']=='lm_head');assert all(expected[key]==value for key,value in state['weight_signature'].items());del wf
  tensors={};guards=[]
  def allocate(name,shape,dtype,value):
   count=int(np.prod(shape));store=torch.full((count+512,),value,dtype=dtype,device='cuda');tensors[name]=store[256:count+256].view(shape);guards.append((store,value));return tensors[name]
  a=allocate('a',(m,k),torch.bfloat16,17.);q=allocate('q',(m,k),torch.int8,17);asc=allocate('asc',(m,),torch.float32,17.);w=allocate('w',(n,k),torch.int8,17);ws=allocate('ws',(n,),torch.float32,17.);integer=allocate('integer',(m,n),torch.int32,-777);reference_int=allocate('reference_integer',(m,n),torch.int32,-777);control_out=allocate('control',(m,n),torch.bfloat16,17.);diag_out=allocate('diagnostic',(m,n),torch.bfloat16,17.);dummy=torch.empty(1,device='cuda',dtype=torch.int32)
  sample=np.random.default_rng(202610221410).standard_normal((m,k),dtype=np.float32);bits=sample.view(np.uint32);base=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);base[0]=0;ac=torch.from_numpy(base).view(torch.bfloat16);a.copy_(ac);w.copy_(codes);ws.copy_(scales)
  def metadata(kernel):return dict(hash=kernel.hash,binary_sha256=hashlib.sha256(kernel.kernel).hexdigest(),n_regs=kernel.n_regs,n_spills=kernel.n_spills,shared=kernel.metadata.shared)
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def control():original.quantize(a,q,asc);return original.mm(q,w,asc,ws,control_out,dummy,config)
  compiled=control();torch.cuda.synchronize();state.update(target=str(triton.runtime.driver.active.get_current_target()),M=m,N=n,K=k,configuration=config,current_compiled=metadata(compiled),INT32_workspace_bytes=m*n*4,variants={});assert compiled.n_regs==93 and compiled.n_spills==0 and compiled.metadata.shared==24576
  operations={'control':control};graphs={};variant_configs={}
  for block in (256,512,1024,2048):
   for warps in (4,8):
    name=f'B{block}_W{warps}';out=allocate(name,(m,n),torch.bfloat16,17.)
    def operation(out=out,block=block,warps=warps):original.quantize(a,q,asc);torch._int_mm(q,w.T,out=integer);return new.dequant(integer,asc,ws,out,block,warps)
    compiled=operation();torch.cuda.synchronize();operations[name]=operation;variant_configs[name]=(block,warps);state['variants'][name]=dict(block=block,warps=warps,compiled=metadata(compiled))
  for name,fn in operations.items():
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):fn()
   graphs[name]=graph
  rows=np.unique(np.linspace(0,m-1,8).astype(np.int64));columns=np.unique(np.concatenate([np.linspace(0,n-1,64).astype(np.int64),np.arange(n-8,n),np.array([0,1,1023,1024,65535,65536])]).astype(np.int64));r_gpu=torch.tensor(rows,device='cuda');c_gpu=torch.tensor(columns,device='cuda');state['phase']='numerical';save()
  for phase in range(3):
   sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.;a.copy_(ac if phase==0 else ac*(.5 if phase==1 else -.5));w.copy_(codes*sign);ws.copy_(scales*factor);immutable=[x.clone() for x in (a,w,ws)];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
   original.quantize(a,q,asc);original.mm(q,w,asc,ws,diag_out,reference_int,config,return_int=True);torch.cuda.synchronize();reference_words=words(diag_out);reference_integer_hash=thash(reference_int.cpu());reference_output_hash=ahash(reference_words);records=[]
   for name,fn in operations.items():
    fn();torch.cuda.synchronize();eager=words(tensors[name]);assert np.array_equal(eager,reference_words)
    tensors[name].fill_(float('nan'));integer.fill_(-2147483647);graphs[name].replay();torch.cuda.synchronize();actual=words(tensors[name]);assert np.array_equal(actual,eager) and np.isfinite(tensors[name].float().cpu().numpy()).all()
    if name!='control':assert torch.equal(integer,reference_int)
    records.append(dict(name=name,output_sha256=ahash(actual),full_BF16_output_exact=True,full_INT32_exact=(name!='control'),graph_poison_matches_eager=True))
   for tensor,before in zip((a,w,ws),immutable):assert torch.equal(tensor,before)
   assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
   for store,value in guards:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
   path=root/f'phase{phase}.npz';np.savez(path,rows=rows,columns=columns,integer_selected=reference_int[r_gpu[:,None],c_gpu[None,:]].cpu().numpy(),output_selected_words=reference_words[np.ix_(rows,columns)])
   state['phases'].append(dict(phase=phase,weight_sign=sign,weight_scale_factor=factor,artifact=path.name,sha256=sha(path),A_words_sha256=ahash(words(a)),Aq_sha256=thash(q.cpu()),As_sha256=thash(asc.cpu()),WS_sha256=thash(ws.cpu()),integer_sha256=reference_integer_hash,output_sha256=reference_output_hash,entries=records,guards_inputs_weights_RNG=True));save()
  state['phase']='timing';save()
  for mode in ('graph','eager'):
   for roundid in range(7):
    names=list(operations);shift=(2*roundid)%len(names);names=names[shift:]+names[:shift]
    if roundid%2:names.reverse()
    timings={}
    for name in names:
     fn=graphs[name].replay if mode=='graph' else operations[name];begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
     for _ in range(20):fn()
     end.record();end.synchronize();timings[name]=begin.elapsed_time(end)/20
    state['rounds'].append(dict(mode=mode,round=roundid,order=names,replays=20,milliseconds=timings));save()
  breakdowns={'quantize':lambda:original.quantize(a,q,asc),'native_int_mm':lambda:torch._int_mm(q,w.T,out=integer),'current_gemm_dequant':lambda:original.mm(q,w,asc,ws,control_out,dummy,config)}
  for name,(block,warps) in variant_configs.items():breakdowns[f'dequant_{name}']=lambda name=name,block=block,warps=warps:new.dequant(integer,asc,ws,tensors[name],block,warps)
  for roundid in range(7):
   names=list(breakdowns);shift=roundid%len(names);names=names[shift:]+names[:shift];timings={}
   for name in names:
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
    for _ in range(20):breakdowns[name]()
    end.record();end.synchronize();timings[name]=begin.elapsed_time(end)/20
   state['breakdowns'].append(dict(round=roundid,repeats=20,milliseconds=timings))
  for store,value in guards:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
  state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_before']==state['checkpoint_sha256_after'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True);save();print(json.dumps(dict(complete=True,variants=len(variant_configs),phases=3)),flush=True)
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
