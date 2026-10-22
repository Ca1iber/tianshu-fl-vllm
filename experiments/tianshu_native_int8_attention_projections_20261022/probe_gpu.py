# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib,importlib.util,inspect,json,os,signal,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261022_native_int8_attention_projections_GPU_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def main():
 remain=int((datetime.fromisoformat('2026-10-22T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert remain>2400
 def stopped(signum,frame):raise TimeoutError(f'signal {signum}')
 signal.signal(signal.SIGALRM,stopped);signal.signal(signal.SIGTERM,stopped);signal.alarm(min(remain,2400));assert not ROOT.exists();ROOT.mkdir();hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(ROOT/'hardware_before.txt').write_text(hardware);sources=dict(read(BASE/'CPU_PREPARE_GATE_32g.json')['sources']);sources[str(BASE/'CPU_PREPARE_GATE_32g.json')]=sha(BASE/'CPU_PREPARE_GATE_32g.json');assert all(sha(p)==h for p,h in sources.items())
 for p,h in sources.items():
  target=ROOT/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,checkpoint_before=sha(MODEL),cases=[],rounds=[],model_executed=False,official_benchmark=False,adopted=False,actual_service_route_not_yet_verified=True)
 def save():
  temp=ROOT/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(ROOT/'result.json')
 save()
 try:
  import torch,vllm_fl,triton,flag_gems
  from safetensors import safe_open
  torch.set_num_threads(4);dense=load('_attention_projection_reference',EXP/'tianshu_dense_w8a8_20261020/kernel.py');dequant=load('_attention_projection_dequant',EXP/'tianshu_native_int8_lm64_20261022/kernel.py');vendor_name=next((n for n in ('_iluvatar.ops.mm','flag_gems.runtime.backend._iluvatar.ops.mm') if n in sys.modules),None);vendor=sys.modules[vendor_name] if vendor_name else importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm');vendor_path=Path(vendor.__file__).resolve();assert sha(vendor_path)=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090' and Path(inspect.getsourcefile(vendor._launch_mm)).resolve()==vendor_path;state.update(vendor_source=str(vendor_path),vendor_source_sha256=sha(vendor_path),target=str(triton.runtime.driver.active.get_current_target()))
  def thash(t):return ahash(t.detach().cpu().contiguous().view(torch.uint8).numpy())
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def pack(t):v=t.float();s=v.abs().amax(1).clamp_min(1.e-8)/127.;return (v/s[:,None]).round().clamp(-127,127).to(torch.int8),s
  def metadata(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared)
  with safe_open(str(MODEL),framework='pt',device='cpu') as model:weights=dict(QKV=torch.cat([model.get_tensor(f'model.layers.0.self_attn.{p}_proj.weight') for p in ('q','k','v')],0).contiguous(),O=model.get_tensor('model.layers.0.self_attn.o_proj.weight').contiguous())
  assert tuple(weights['QKV'].shape)==(2560,2048) and tuple(weights['O'].shape)==(2048,2048)
  def run_case(role,m):
   weight=weights[role];n,k=weight.shape;code,scale=pack(weight);guards=[]
   def allocate(shape,dtype,value):
    count=int(np.prod(shape));store=torch.full((count+512,),value,dtype=dtype,device='cuda');guards.append((store,value));return store[256:count+256].view(shape)
   x=allocate((m,k),torch.bfloat16,17.);q=allocate((m,k),torch.int8,17);asc=allocate((m,),torch.float32,17.);wd=allocate((n,k),torch.bfloat16,17.);wnt=allocate((n,k),torch.int8,17);wnn=allocate((k,n),torch.int8,17);ws=allocate((n,),torch.float32,17.);integer=allocate((m,n),torch.int32,-777);refint=allocate((m,n),torch.int32,-777);refout=allocate((m,n),torch.bfloat16,17.);names=['control']+[f'{layout}_B{block}' for layout in ('NT','NN') for block in (512,1024)];outputs={name:allocate((m,n),torch.bfloat16,17.) for name in names};sample=np.random.default_rng(202610232000+m+(0 if role=='QKV' else 10)).standard_normal((m,k),dtype=np.float32);bits=sample.view(np.uint32);bits=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);bits[0]=0;base=torch.from_numpy(bits).view(torch.bfloat16)
   def fixture(phase):
    sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.;x.copy_(base if phase==0 else base*(.5 if phase==1 else -.5));wd.copy_(weight*sign*factor);wnt.copy_(code*sign);wnn.copy_((code*sign).T.contiguous());ws.copy_(scale*factor)
   def control():return vendor._launch_mm(x,wd.T,outputs['control'],m,n,k)
   operations=dict(control=control);fixture(0);compiled={}
   for layout in ('NT','NN'):
    for block in (512,1024):
     name=f'{layout}_B{block}';packed=wnt.T if layout=='NT' else wnn;out=outputs[name]
     def operation(packed=packed,out=out,block=block):dense.quantize(x,q,asc);torch._int_mm(q,packed,out=integer);return dequant._dequant[(triton.cdiv(m*n,block),)](integer,asc,ws,out,m,n,block,num_warps=4,num_stages=1)
     operations[name]=operation;compiled[name]=metadata(operation())
   control();torch.cuda.synchronize();graphs={}
   for name,fn in operations.items():
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn()
    graphs[name]=graph
   case=dict(role=role,M=m,N=n,K=k,weight_signature=dict(original_BF16_sha256=thash(weight),packed_sha256=thash(code),scales_sha256=thash(scale)),packed_payload_bytes=code.numel()+scale.numel()*4,INT32_workspace_bytes=m*n*4,compiled=compiled,complete=False,phases=[]);state['cases'].append(case);state['phase']=f'{role}_M{m}';save();rows=np.unique(np.linspace(0,m-1,8).astype(np.int64));cols=np.unique(np.concatenate([np.linspace(0,n-1,32).astype(np.int64),np.arange(n-8,n)]));rg=torch.tensor(rows,device='cuda');cg=torch.tensor(cols,device='cuda')
   for phase in range(3):
    fixture(phase);immutable=[t.clone() for t in (x,wd,wnt,wnn,ws)];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();dense.quantize(x,q,asc);dense.mm(q,wnt,asc,ws,refout,refint,dict(bm=32,bn=128,bk=128,warps=2,stages=1),True);torch.cuda.synchronize();reference=words(refout);entries=[]
    for name,fn in operations.items():
     fn();torch.cuda.synchronize();out=outputs[name];eager=words(out);out.fill_(float('nan'));integer.fill_(-2147483647);graphs[name].replay();torch.cuda.synchronize();assert np.array_equal(words(out),eager) and bool(torch.isfinite(out).all()) and bool((out[0]==0).all())
     if name!='control':assert torch.equal(integer,refint) and np.array_equal(eager,reference)
     entries.append(dict(name=name,full_INT32_exact=(name!='control'),full_quantized_BF16_exact=(name!='control'),output_sha256=ahash(eager),graph_poison_eager_exact=True))
    relative=float(torch.linalg.vector_norm(refout.float()-outputs['control'].float())/torch.linalg.vector_norm(outputs['control'].float()).clamp_min(1e-20));assert all(torch.equal(a,b) for a,b in zip((x,wd,wnt,wnn,ws),immutable)) and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    for store,value in guards:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
    path=ROOT/f'{role}_M{m}_phase{phase}.npz';np.savez(path,X_words=words(x),rows=rows,columns=cols,integer_selected=refint[rg[:,None],cg[None,:]].cpu().numpy(),candidate_selected_words=reference[np.ix_(rows,cols)],control_selected_words=words(outputs['control'][rg[:,None],cg[None,:]]));case['phases'].append(dict(phase=phase,artifact=path.name,sha256=sha(path),X_words_sha256=ahash(words(x)),Aq_sha256=thash(q),As_sha256=thash(asc),WS_sha256=thash(ws),reference_output_sha256=ahash(reference),relative_L2_vs_BF16=relative,guards_inputs_weights_RNG=True,entries=entries));save()
   for mode in ('graph','eager'):
    for roundid in range(7):
     order=list(operations);shift=roundid%len(order);order=order[shift:]+order[:shift]
     if roundid%2:order.reverse()
     times={}
     for name in order:
      fn=graphs[name].replay if mode=='graph' else operations[name];begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
      for _ in range(10):fn()
      end.record();end.synchronize();times[name]=begin.elapsed_time(end)/10
     state['rounds'].append(dict(role=role,M=m,mode=mode,round=roundid,order=order,replays=10,milliseconds=times))
   case['complete']=True;save();print(json.dumps(dict(role=role,M=m,relative_L2=relative,complete=True)),flush=True)
  for role in ('QKV','O'):
   for m in (64,128,512,4096):run_case(role,m);torch.cuda.empty_cache()
  state['checkpoint_after']=sha(MODEL);assert state['checkpoint_after']==state['checkpoint_before'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:signal.alarm(0);(ROOT/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
