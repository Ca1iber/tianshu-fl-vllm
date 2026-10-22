# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib,inspect,json,os,signal,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261022_down_BF16_control_GPU_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
 remain=int((datetime.fromisoformat('2026-10-22T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert remain>900
 def stop(signum,frame):raise TimeoutError(str(signum))
 signal.signal(signal.SIGALRM,stop);signal.alarm(min(remain,600));assert not ROOT.exists();ROOT.mkdir();hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(ROOT/'hardware_before.txt').write_text(hardware);state=dict(complete=False,phase='initializing',baseline_diagnostic_only=True,model_executed=False,official_benchmark=False,adopted=False,rounds=[],checkpoint_before=sha(MODEL))
 def save():
  temp=ROOT/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(ROOT/'result.json')
 save()
 try:
  import torch,vllm_fl,flag_gems
  from safetensors import safe_open
  torch.set_num_threads(4);name=next((s for s in ('_iluvatar.ops.mm','flag_gems.runtime.backend._iluvatar.ops.mm') if s in sys.modules),None);module=sys.modules[name] if name else importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm');module_path=Path(module.__file__).resolve();assert sha(module_path)=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090';hook=EXP/'tianshu_prefill_gemm_tiles_20261014/hooks/sitecustomize.py';assert sha(hook)=='eb2e24f0bb1cf1cade2474075404cda46aac05a5b34eee45d42b133fa48c6928' and 'M!=2048' in hook.read_text();launch_path=Path(inspect.getsourcefile(module._launch_mm)).resolve();assert launch_path==module_path
  gate=json.loads((EXP/'tianshu_native_int8_down_prefill_official_results_v2_20261022/OFFICIAL_GATE_32g.json').read_text());assert gate['promotion_passed'] and gate['model_frozen_sources'][str(module_path)]==sha(module_path) and gate['model_frozen_sources'][str(hook)]==sha(hook)
  sources={str(p):sha(p) for p in (BASE/'probe_gpu.py',BASE/'PLAN_32g.md',module_path,hook,EXP/'tianshu_native_int8_down_prefill_official_results_v2_20261022/OFFICIAL_GATE_32g.json')};state.update(frozen_sources=sources,vendor_source=str(module_path),vendor_source_sha256=sha(module_path),launch_source=str(launch_path),M64_service_wrapper_delegates_to_this_source=True,torch_version=torch.__version__)
  for p,h in sources.items():
   target=ROOT/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
  with safe_open(str(MODEL),framework='pt',device='cpu') as model:weight=model.get_tensor('model.layers.0.mlp.down_proj.weight').contiguous()
  sample=np.random.default_rng(202610230100).standard_normal((64,6144),dtype=np.float32);bits=sample.view(np.uint32);bits=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);bits[0]=0;x=torch.from_numpy(bits).view(torch.bfloat16).cuda();w=weight.cuda();output={k:torch.empty((64,2048),dtype=torch.bfloat16,device='cuda') for k in ('torch_mm','service_vendor_launch')}
  def thash(t):return ahash(t.cpu().contiguous().view(torch.uint8).numpy())
  x_before,w_before=thash(x),thash(w);cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();operations={'torch_mm':lambda:torch.mm(x,w.T,out=output['torch_mm']),'service_vendor_launch':lambda:module._launch_mm(x,w.T,output['service_vendor_launch'],64,2048,6144)}
  for fn in operations.values():fn()
  direct=module.mm(x,w.T);torch.cuda.synchronize();assert torch.equal(direct,output['service_vendor_launch']);hashes={k:thash(t) for k,t in output.items()};graphs={}
  for name,fn in operations.items():
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):fn()
   graphs[name]=graph;output[name].fill_(float('nan'));graph.replay();torch.cuda.synchronize();assert thash(output[name])==hashes[name] and bool(torch.isfinite(output[name]).all())
  for index in range(7):
   order=list(operations) if index%2==0 else list(reversed(operations));times={}
   for name in order:
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
    for _ in range(100):graphs[name].replay()
    end.record();end.synchronize();times[name]=begin.elapsed_time(end)/100
   state['rounds'].append(dict(round=index,order=order,replays=100,milliseconds=times))
  a,b=output.values();difference=(a.float()-b.float()).abs();state.update(full_BF16_equal=bool(torch.equal(a,b)),different_output_cells=int((a!=b).sum()),output_relative_L2=float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(a.float())),output_max_abs_difference=float(difference.max()))
  assert thash(x)==x_before and thash(w)==w_before and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());state['checkpoint_after']=sha(MODEL);assert state['checkpoint_before']==state['checkpoint_after'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete',input_weight_RNG_source_checkpoint_guards=True,medians={k:float(np.median([r['milliseconds'][k] for r in state['rounds']])) for k in operations});save();print(json.dumps({k:v for k,v in state.items() if k not in ('frozen_sources','rounds')}),flush=True)
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
