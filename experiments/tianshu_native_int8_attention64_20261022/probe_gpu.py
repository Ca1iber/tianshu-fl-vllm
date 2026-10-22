# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib,importlib.util,inspect,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261022_native_int8_attention64_GPU_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def main():
 remain=int((datetime.fromisoformat('2026-10-22T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert remain>2400
 def stopped(signum,frame):raise TimeoutError(str(signum))
 signal.signal(signal.SIGALRM,stopped);signal.signal(signal.SIGTERM,stopped);signal.alarm(min(remain,2400));assert not ROOT.exists();ROOT.mkdir();hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(ROOT/'hardware_before.txt').write_text(hardware);sources=dict(read(BASE/'CPU_PREPARE_GATE_32g.json')['sources']);sources[str(BASE/'CPU_PREPARE_GATE_32g.json')]=sha(BASE/'CPU_PREPARE_GATE_32g.json');assert all(sha(p)==h for p,h in sources.items())
 for p,h in sources.items():
  target=ROOT/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,checkpoint_before=sha(MODEL),banks=[],cases=[],rounds=[],model_executed=False,official_benchmark=False,adopted=False,actual_service_route_not_yet_verified=True,combined_pair_is_not_attention=True)
 def save():
  temp=ROOT/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(ROOT/'result.json')
 save()
 try:
  import torch,vllm_fl,triton,flag_gems
  from safetensors import safe_open
  from vllm.model_executor.layers.linear import UnquantizedLinearMethod
  from vllm_fl.utils import get_flag_gems_whitelist_blacklist
  torch.set_num_threads(4);dense=load('_attention42_ref',EXP/'tianshu_dense_w8a8_20261020/kernel.py');dequant=load('_attention42_dequant',EXP/'tianshu_native_int8_lm64_20261022/kernel.py');vendor_name=next((n for n in ('_iluvatar.ops.mm','flag_gems.runtime.backend._iluvatar.ops.mm') if n in sys.modules),None);vendor=sys.modules[vendor_name] if vendor_name else importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm');vendor_path=Path(vendor.__file__).resolve();assert sha(vendor_path)=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090' and Path(inspect.getsourcefile(vendor._launch_mm)).resolve()==vendor_path;state['target']=str(triton.runtime.driver.active.get_current_target())
  def thash(t):return ahash(t.detach().cpu().contiguous().view(torch.uint8).numpy())
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def pack(t):v=t.float();s=v.abs().amax(1).clamp_min(1.e-8)/127.;return (v/s[:,None]).round().clamp(-127,127).to(torch.int8),s
  def metadata(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared)
  banks=[]
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint:
   for layer in range(42):
    for role in ('QKV','O'):
     weight=torch.cat([checkpoint.get_tensor(f'model.layers.{layer}.self_attn.{p}_proj.weight') for p in ('q','k','v')],0).contiguous() if role=='QKV' else checkpoint.get_tensor(f'model.layers.{layer}.self_attn.o_proj.weight').contiguous();q,s=pack(weight);proof=dict(layer=layer,role=role,prefix=f'model.layers.{layer}.self_attn.'+('qkv_proj' if role=='QKV' else 'o_proj'),N=weight.shape[0],K=weight.shape[1],original_BF16_sha256=thash(weight),packed_sha256=thash(q),scales_sha256=thash(s),packed_payload_bytes=q.numel()+s.numel()*4);banks.append(dict(W=weight.cuda(),Q=q.T.contiguous().cuda(),S=s.cuda()));state['banks'].append(proof)
    print(json.dumps(dict(layer=layer,loaded=True)),flush=True)
  state.update(phase='banks_loaded',new_packed_payload_bytes=sum(b['packed_payload_bytes'] for b in state['banks']));save()
  whitelist,blacklist=get_flag_gems_whitelist_blacklist();assert not whitelist and not blacklist;linear_path=Path(inspect.getsourcefile(flag_gems.linear)).resolve();assert sources[str(linear_path)]==sha(linear_path);flag_gems.enable(record=False,once=True);method=UnquantizedLinearMethod();layers=[SimpleNamespace(weight=b['W'],prefix=p['prefix']) for p,b in zip(state['banks'],banks)];state.update(BF16_control='actual_UnquantizedLinearMethod_F_linear',actual_linear_source=str(linear_path),actual_linear_sha256=sha(linear_path))
  def run_case(m):
   guards=[]
   def allocate(shape,dtype,value):
    count=int(np.prod(shape));store=torch.full((count+512,),value,dtype=dtype,device='cuda');guards.append((store,value));return store[256:count+256].view(shape)
   x=allocate((m,2048),torch.bfloat16,17.);q=allocate((m,2048),torch.int8,17);asc=allocate((m,),torch.float32,17.);integer={n:allocate((m,n),torch.int32,-777) for n in (2048,2560)};refint={n:allocate((m,n),torch.int32,-777) for n in (2048,2560)};refout={n:allocate((m,n),torch.bfloat16,17.) for n in (2048,2560)};outputs={mode:[allocate((m,p['N']),torch.bfloat16,17.) for p in state['banks']] for mode in ('control','candidate')};block=512
   def fixture(phase):
    sample=np.random.default_rng(202610237200+m+phase).standard_normal((m,2048),dtype=np.float32);bits=sample.view(np.uint32);bits=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);bits[0]=0;x.copy_(torch.from_numpy(bits).view(torch.bfloat16))
   def control(index):out=method.apply(layers[index],x,None);outputs['control'][index]=out;return out
   def candidate(index):
    bank=banks[index];n=bank['W'].shape[0];dense.quantize(x,q,asc);torch._int_mm(q,bank['Q'],out=integer[n]);return dequant._dequant[(triton.cdiv(m*n,block),)](integer[n],asc,bank['S'],outputs['candidate'][index],m,n,block,num_warps=4,num_stages=1)
   fixture(0);compiled={}
   for index,proof in enumerate(state['banks']):control(index);c=candidate(index);compiled.setdefault(proof['role'],metadata(c))
   torch.cuda.synchronize();graphs={};graph_outputs={}
   for scope in ('QKV','O','combined_pair'):
    indices=[i for i,b in enumerate(state['banks']) if scope=='combined_pair' or b['role']==scope]
    for mode,operation in (('control',control),('candidate',candidate)):
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):
      for index in indices:operation(index)
     graphs[(scope,mode)]=graph;graph_outputs[(scope,mode)]=list(outputs[mode])
   case=dict(M=m,block=block,compiled=compiled,complete=False,phases=[]);state['cases'].append(case);state['phase']=f'M{m}';save();rows=np.unique(np.linspace(0,m-1,8).astype(np.int64));rg=torch.tensor(rows,device='cuda')
   for phase in range(2):
    fixture(phase);before=x.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();records=[];hashes={mode:[] for mode in outputs}
    for index,(proof,bank) in enumerate(zip(state['banks'],banks)):
     n=proof['N'];cols=np.unique(np.linspace(0,n-1,32).astype(np.int64));cg=torch.tensor(cols,device='cuda');control(index);candidate(index);torch.cuda.synchronize();dense.quantize(x,q,asc);dense.mm(q,bank['Q'].T.contiguous(),asc,bank['S'],refout[n],refint[n],dict(bm=32,bn=128,bk=128,warps=2,stages=1),True);torch.cuda.synchronize();assert torch.equal(integer[n],refint[n]) and torch.equal(outputs['candidate'][index],refout[n]);error=outputs['candidate'][index].float()-outputs['control'][index].float();relative=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(outputs['control'][index].float()).clamp_min(1e-20))
     for mode in outputs:assert bool(torch.isfinite(outputs[mode][index]).all());hashes[mode].append(ahash(words(outputs[mode][index])))
     path=ROOT/f'M{m}_phase{phase}_{proof["role"]}_layer{proof["layer"]}.npz';np.savez(path,rows=rows,columns=cols,X_selected_words=words(x[rg]),Aq_selected=q[rg].cpu().numpy(),As_selected=asc[rg].cpu().numpy(),integer_selected=refint[n][rg[:,None],cg[None,:]].cpu().numpy(),candidate_selected_words=words(outputs['candidate'][index][rg[:,None],cg[None,:]]),control_selected_words=words(outputs['control'][index][rg[:,None],cg[None,:]]));records.append(dict(index=index,role=proof['role'],layer=proof['layer'],artifact=path.name,sha256=sha(path),full_INT32_BF16_exact=True,relative_L2_vs_BF16=relative))
    for mode in outputs:
     for out in graph_outputs[('combined_pair',mode)]:out.fill_(float('nan'))
     graphs[('combined_pair',mode)].replay()
    torch.cuda.synchronize()
    for mode in outputs:
     for index,out in enumerate(graph_outputs[('combined_pair',mode)]):assert bool(torch.isfinite(out).all()) and ahash(words(out))==hashes[mode][index]
    assert torch.equal(x,before) and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    for store,value in guards:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
    case['phases'].append(dict(phase=phase,seed=202610237200+m+phase,banks=records,whole84_graph_poison_eager_exact=True,inputs_RNG_canaries_unchanged=True,output_hashes=hashes));save();print(json.dumps(dict(M=m,phase=phase,all84=True)),flush=True)
   for scope in ('QKV','O','combined_pair'):
    for roundid in range(7):
     order=['control','candidate'] if roundid%2==0 else ['candidate','control'];times={}
     for mode in order:
      begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
      for _ in range(3):graphs[(scope,mode)].replay()
      end.record();end.synchronize();times[mode]=begin.elapsed_time(end)/3/42
     state['rounds'].append(dict(M=m,scope=scope,round=roundid,order=order,replays=3,layers_per_graph=42,milliseconds_per_layer=times))
   case['complete']=True;save()
  for m in (64,):run_case(m);torch.cuda.empty_cache()
  for proof,bank in zip(state['banks'],banks):assert thash(bank['W'])==proof['original_BF16_sha256'] and thash(bank['Q'].T.contiguous())==proof['packed_sha256'] and thash(bank['S'])==proof['scales_sha256']
  state['checkpoint_after']=sha(MODEL);assert state['checkpoint_after']==state['checkpoint_before'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True,all84_bank_hashes_before_after=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:signal.alarm(0);(ROOT/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
