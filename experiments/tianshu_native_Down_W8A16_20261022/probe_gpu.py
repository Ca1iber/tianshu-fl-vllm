# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib,importlib.util,inspect,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261022_native_Down_W8A16_GPU_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def main():
 remain=int((datetime.fromisoformat('2026-10-22T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert remain>1800
 def stopped(signum,frame):raise TimeoutError(f'signal {signum}')
 signal.signal(signal.SIGALRM,stopped);signal.signal(signal.SIGTERM,stopped);signal.alarm(min(remain,1800));assert not ROOT.exists();ROOT.mkdir();hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(ROOT/'hardware_before.txt').write_text(hardware)
 sources=dict(read(BASE/'CPU_PREPARE_GATE_32g.json')['sources']);sources[str(BASE/'CPU_PREPARE_GATE_32g.json')]=sha(BASE/'CPU_PREPARE_GATE_32g.json');assert all(sha(p)==h for p,h in sources.items())
 for p,h in sources.items():
  if Path(p).suffix=='.so':continue
  target=ROOT/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,model_executed=False,official_benchmark=False,adopted=False,single_bank_cache_screen_only=True,layer=0,checkpoint_before=sha(MODEL),rounds=[],phases=[])
 def save():
  temp=ROOT/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(ROOT/'result.json')
 save()
 try:
  import torch,vllm_fl,triton,flag_gems
  from safetensors import safe_open
  torch.set_num_threads(4);quant=load('_w8a16_GU_quant',EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py');fused=load('_w8a16_GU64',EXP/'tianshu_nn_silu_two_dot_20261021/kernel.py');selection=next(r for r in read(EXP/'tianshu_nn_silu_two_dot_20261021/GPU_GATE_32g.json')['best_by_shape'] if r['M']==64 and not r['strided']);expected_gu=selection['actual_timed_compiled']['binary_sha256']
  vendor_name=next((n for n in ('_iluvatar.ops.mm','flag_gems.runtime.backend._iluvatar.ops.mm') if n in sys.modules),None);vendor=sys.modules[vendor_name] if vendor_name else importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm');vendor_path=Path(vendor.__file__).resolve();assert sha(vendor_path)=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090';assert Path(inspect.getsourcefile(vendor._launch_mm)).resolve()==vendor_path
  launch_hook=EXP/'tianshu_prefill_gemm_tiles_20261014/hooks/sitecustomize.py';assert sources[str(launch_hook)]==sha(launch_hook)=='eb2e24f0bb1cf1cade2474075404cda46aac05a5b34eee45d42b133fa48c6928' and 'M!=2048' in launch_hook.read_text();state['actual_service_M64_vendor_delegation_verified']=True
  def thash(t):return ahash(t.detach().cpu().contiguous().view(torch.uint8).numpy())
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def metadata(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared)
  def pack(t):v=t.float();s=v.abs().amax(1).clamp_min(1.e-8)/127.;return (v/s[:,None]).round().clamp(-127,127).to(torch.int8),s
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint:gu=torch.cat([checkpoint.get_tensor(f'model.layers.0.mlp.{role}_proj.weight') for role in ('gate','up')],0).contiguous();down=checkpoint.get_tensor('model.layers.0.mlp.down_proj.weight').contiguous()
  guq,gus=pack(gu);dq,ds=pack(down);proof=dict(GU_original_BF16_sha256=thash(gu),GU_packed_sha256=thash(guq),GU_scales_sha256=thash(gus),Down_original_BF16_sha256=thash(down),Down_packed_sha256=thash(dq),Down_scales_sha256=thash(ds));old=read(Path(read(EXP/'tianshu_native_int8_down42_20261022/INDEPENDENT_CPU_GATE_32g.json')['raw_directory'])/'result.json')['banks'][0];assert all(proof[k]==old[k] for k in proof);state.update(bank=proof,target=str(triton.runtime.driver.active.get_current_target()),GateUp_binary_sha256=expected_gu,vendor_source=str(vendor_path),vendor_source_sha256=sha(vendor_path))
  guw,guscale=guq.T.contiguous().cuda(),gus.cuda();wd,wnt,wnn,ws=down.cuda(),dq.cuda(),dq.T.contiguous().cuda(),ds.cuda();guards=[]
  def allocate(shape,dtype,value):
   count=int(np.prod(shape));store=torch.full((count+512,),value,dtype=dtype,device='cuda');guards.append((store,value));return store[256:count+256].view(shape)
  m=64;a=allocate((m,2048),torch.bfloat16,17.);aq=allocate((m,2048),torch.int8,17);asc=allocate((m,),torch.float32,17.);h=allocate((m,6144),torch.bfloat16,17.);dummy=torch.empty(1,device='cuda',dtype=torch.int32)
  configurations=[dict(layout=l,name=f'native_W8A16_{l}') for l in ('NT','NN')];outputs={}
  def fixture(phase):
   sample=np.random.default_rng(202610239200+phase).standard_normal((m,2048),dtype=np.float32);bits=sample.view(np.uint32);bits=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);bits[0]=0;a.copy_(torch.from_numpy(bits).view(torch.bfloat16))
  def gateup():quant.quantize(a,aq,asc);return fused.fused(aq,guw,asc,guscale,h,dummy,dummy.view(torch.bfloat16),selection['configuration'])
  def control():out=vendor.mm(h,wd.T);outputs['service_BF16']=out;return out
  def secondary():out=torch.mm(h,wd.T);outputs['torch_BF16']=out;return out
  operations=dict(service_BF16=control,torch_BF16=secondary);fixture(0);assert metadata(gateup())['binary_sha256']==expected_gu;state['compiled']={};state['configurations']=configurations
  for configuration in configurations:
   name=configuration['name']
   def operation(configuration=configuration,name=name):out=torch.ops.aten._weight_int8pack_mm.default(h,wnn.T if configuration['layout']=='NN' else wnt,ws);outputs[name]=out;return out
   operations[name]=operation;operation();state['compiled'][name]=dict(registered_op='aten::_weight_int8pack_mm',layout=configuration['layout'],BF16_input_output=True,FP32_scales=True)
  for fn in operations.values():fn()
  assert torch.equal(vendor.mm(h,wd.T),outputs['service_BF16']);torch.cuda.synchronize();graphs={};graph_outputs={}
  for scope in ('whole_MLP','Down_only'):
   for name,fn in operations.items():
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     if scope=='whole_MLP':gateup()
     fn()
    graph_outputs[(scope,name)]=outputs[name]
    graphs[(scope,name)]=graph
  state['phase']='validation';save()
  for phase in range(2):
   fixture(phase);gateup();before=a.clone();hidden=h.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();artifact=dict(A_words=words(a),H_words=words(h),Down_packed=dq.numpy(),Down_scales=ds.numpy());hashes={};relative={}
   for name,fn in operations.items():
    fn();torch.cuda.synchronize();out=outputs[name];assert bool(torch.isfinite(out).all()) and bool((out[0]==0).all());artifact[name+'_words']=words(out);hashes[name]=ahash(artifact[name+'_words'])
    if name!='service_BF16':relative[name]=float(torch.linalg.vector_norm(out.float()-outputs['service_BF16'].float())/torch.linalg.vector_norm(outputs['service_BF16'].float()).clamp_min(1e-20))
    graph_outputs[('whole_MLP',name)].fill_(float('nan'));h.fill_(float('nan'));graphs[('whole_MLP',name)].replay();torch.cuda.synchronize();assert ahash(words(graph_outputs[('whole_MLP',name)]))==hashes[name]
   assert torch.equal(a,before) and torch.equal(h,hidden) and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
   for store,value in guards:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
   path=ROOT/f'M64_phase{phase}.npz';np.savez(path,**artifact);state['phases'].append(dict(phase=phase,seed=202610239200+phase,artifact=path.name,sha256=sha(path),output_hashes=hashes,graph_poison_eager_exact=True,guards_inputs_weights_RNG=True,relative_L2_vs_service_BF16=relative));save()
  state['phase']='timing';save()
  for scope in ('whole_MLP','Down_only'):
   for mode in ('graph','eager'):
    for roundid in range(7):
     order=list(operations);shift=roundid%len(order);order=order[shift:]+order[:shift]
     if roundid%2:order.reverse()
     times={}
     for name in order:
      fn=graphs[(scope,name)].replay if mode=='graph' else operations[name];begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
      for _ in range(20):
       if mode=='eager' and scope=='whole_MLP':gateup()
       fn()
      end.record();end.synchronize();times[name]=begin.elapsed_time(end)/20
     state['rounds'].append(dict(M=m,scope=scope,mode=mode,round=roundid,order=order,replays=20,milliseconds=times))
  assert thash(guw.T.contiguous())==proof['GU_packed_sha256'] and thash(guscale)==proof['GU_scales_sha256'] and thash(wd)==proof['Down_original_BF16_sha256'] and thash(wnt)==proof['Down_packed_sha256'] and thash(wnn.T.contiguous())==proof['Down_packed_sha256'] and thash(ws)==proof['Down_scales_sha256']
  for store,value in guards:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
  state['checkpoint_after']=sha(MODEL);assert state['checkpoint_after']==state['checkpoint_before'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True,all_bank_hashes_before_after=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:signal.alarm(0);(ROOT/'hardware_after_internal.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
