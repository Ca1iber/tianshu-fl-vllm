# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib,inspect,json,os,signal,statistics,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261022_attention_BF16_route_GPU_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
 remain=int((datetime.fromisoformat('2026-10-22T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert remain>1200
 def stopped(signum,frame):raise TimeoutError(str(signum))
 signal.signal(signal.SIGALRM,stopped);signal.signal(signal.SIGTERM,stopped);signal.alarm(min(remain,1200));assert not ROOT.exists();ROOT.mkdir();hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(ROOT/'hardware_before.txt').write_text(hardware);profile=json.loads((EXP/'tianshu_native_int8_down_prefill_release_v2_20261022/serve_profile_template.json').read_text());sources=dict(profile['frozen_sources']);assert len(sources)==890 and all(sha(p)==h for p,h in sources.items());sources.update({str(p):sha(p) for p in (BASE/'probe_gpu.py',BASE/'PLAN_32g.md')})
 for p,h in sources.items():
  target=ROOT/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',sources=sources,checkpoint_before=sha(MODEL),model_executed=False,official_benchmark=False,adopted=False,baseline_dispatch_diagnostic=True,full_compiled_service_not_yet_verified=True,cases=[],rounds=[])
 def save():
  temp=ROOT/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(ROOT/'result.json')
 save()
 try:
  import torch,vllm_fl,flag_gems
  from safetensors import safe_open
  from vllm_fl.utils import get_flag_gems_whitelist_blacklist
  from vllm.model_executor.layers.linear import UnquantizedLinearMethod
  torch.set_num_threads(4);whitelist,blacklist=get_flag_gems_whitelist_blacklist();assert not whitelist and not blacklist;vendor_name=next((n for n in ('_iluvatar.ops.mm','flag_gems.runtime.backend._iluvatar.ops.mm') if n in sys.modules),None);vendor=sys.modules[vendor_name] if vendor_name else importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm');assert sha(vendor.__file__)=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090';linear_path=Path(inspect.getsourcefile(flag_gems.linear)).resolve();assert str(linear_path) in sources and sha(linear_path)==sources[str(linear_path)];flag_gems.enable(record=False,once=True);state.update(flag_gems_linear_source=str(linear_path),flag_gems_linear_sha256=sha(linear_path),registration='enable_all_same_as_WorkerFL_empty_blacklist',unquantized_apply_source=inspect.getsourcefile(UnquantizedLinearMethod.apply));method=UnquantizedLinearMethod()
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def thash(t):return ahash(t.detach().cpu().contiguous().view(torch.uint8).numpy())
  with safe_open(str(MODEL),framework='pt',device='cpu') as model:weights=dict(QKV=torch.cat([model.get_tensor(f'model.layers.0.self_attn.{p}_proj.weight') for p in ('q','k','v')],0).contiguous(),O=model.get_tensor('model.layers.0.self_attn.o_proj.weight').contiguous())
  for role in ('QKV','O'):
   weight=weights[role].cuda();n,k=weight.shape;weight_hash=thash(weight);layer=SimpleNamespace(weight=weight,prefix=f'model.layers.0.self_attn.{role}')
   for m in (64,128,512,4096):
    x=torch.empty((m,k),device='cuda',dtype=torch.bfloat16);explicit=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
    def fixture(phase):
     sample=np.random.default_rng(202610233700+m+phase).standard_normal((m,k),dtype=np.float32);bits=sample.view(np.uint32);bits=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);bits[0]=0;x.copy_(torch.from_numpy(bits).view(torch.bfloat16))
    def actual():return method.apply(layer,x,None)
    def control():vendor._launch_mm(x,weight.T,explicit,m,n,k);return explicit
    fixture(0);traces=[]
    def trace(frame,event,arg):
     if event=='call' and 'flag_gems' in frame.f_code.co_filename and frame.f_code.co_name in ('linear','mm','_launch_mm'):traces.append(dict(function=frame.f_code.co_name,source=frame.f_code.co_filename,line=frame.f_code.co_firstlineno))
    sys.setprofile(trace)
    try:actual()
    finally:sys.setprofile(None)
    control();torch.cuda.synchronize();graphs={};graph_outputs={};operations=dict(actual_unquantized_apply=actual,explicit_vendor_mm=control)
    for name,fn in operations.items():
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):out=fn()
     graphs[name]=graph;graph_outputs[name]=out
    case=dict(role=role,M=m,N=n,K=k,python_warm_call_trace=traces,phases=[]);state['cases'].append(case)
    for phase in range(2):
     fixture(phase);before=x.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();values={}
     for name,fn in operations.items():
      eager=fn();torch.cuda.synchronize();expected=words(eager);graph_outputs[name].fill_(float('nan'));graphs[name].replay();torch.cuda.synchronize();assert np.array_equal(words(graph_outputs[name]),expected) and bool(torch.isfinite(graph_outputs[name]).all()) and bool((graph_outputs[name][0]==0).all());values[name]=expected
     assert torch.equal(x,before) and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state()) and thash(weight)==weight_hash;rows=np.unique(np.linspace(0,m-1,8).astype(np.int64));cols=np.unique(np.linspace(0,n-1,32).astype(np.int64));path=ROOT/f'{role}_M{m}_phase{phase}.npz';np.savez(path,rows=rows,columns=cols,X_words=words(x)[rows],**{name+'_words':v[np.ix_(rows,cols)] for name,v in values.items()});case['phases'].append(dict(phase=phase,artifact=path.name,sha256=sha(path),different_full_BF16_cells=int(np.count_nonzero(values['actual_unquantized_apply']!=values['explicit_vendor_mm'])),full_output_cells=m*n,graph_eager_poison_exact=True,input_weight_RNG_unchanged=True))
    for mode in ('graph','eager'):
     for roundid in range(7):
      order=list(operations) if roundid%2==0 else list(reversed(operations));times={}
      for name in order:
       fn=graphs[name].replay if mode=='graph' else operations[name];begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
       for _ in range(20):fn()
       end.record();end.synchronize();times[name]=begin.elapsed_time(end)/20
      state['rounds'].append(dict(role=role,M=m,mode=mode,round=roundid,order=order,milliseconds=times))
    save();print(json.dumps(dict(role=role,M=m,traces=traces)),flush=True)
  state['checkpoint_after']=sha(MODEL);assert state['checkpoint_before']==state['checkpoint_after'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete');save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:signal.alarm(0);(ROOT/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
