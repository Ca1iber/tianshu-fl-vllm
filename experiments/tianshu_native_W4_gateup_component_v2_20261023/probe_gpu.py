# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 signal.alarm(900);assert time.time()<1792771080-900
 g=json.loads((BASE/'CPU_GATE_32g.json').read_text());root=Path(g['root']);out=root/'GPU_v2';assert not out.exists();out.mkdir();assert all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items())
 hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(out/'hardware_before.txt').write_text(hw)
 state=dict(phase='initializing',complete=False,pid=os.getpid(),numeric=[],rounds=[],CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),sources=g['sources'],adopted=False,model_quality_tested=False,official_benchmark=False)
 def save():(out/'result.json').write_text(json.dumps(state,indent=2)+'\n')
 save()
 try:
  import torch,flag_gems
  from flag_gems.fused.silu_and_mul import silu_and_mul_out
  torch.set_num_threads(4);native=load('_gateup_real_quantizer',EXP/'tianshu_int8_weight_nn_20261021/kernel_nn.py');fused=load('_gateup_real_fused_baseline',EXP/'tianshu_nn_silu_two_dot_20261021/kernel.py')
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def tensor(a):
   t=torch.from_numpy(a.copy());return (t.view(torch.bfloat16) if a.dtype==np.uint16 else t).to('cuda').contiguous()
  cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
  for bank in g['banks']:
   pool=torch.cuda.graph_pool_handle();layer=bank['layer'];arrays=dict(np.load(bank['file'],allow_pickle=False));t={k:tensor(v) for k,v in arrays.items()}
   with np.load(root/'X0.npz') as f:x=tensor(f['X'])
   q=torch.empty_like(x,dtype=torch.int8);sa=torch.empty(64,device='cuda',dtype=torch.float32);o=torch.empty((64,6144),device='cuda',dtype=torch.bfloat16);co=torch.empty_like(o);dummy=torch.empty(1,device='cuda',dtype=torch.int32)
   immutable={k:(int(v._version),words(v) if v.dtype==torch.bfloat16 else v.cpu().numpy().copy()) for k,v in t.items()}
   def control():
    native.quantize(x,q,sa);fused.fused(q,t['W8'],sa,t['S8'],o,dummy,dummy.view(torch.bfloat16),g['baseline_configuration']);return o
   def candidate():
    y=torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,t['W4'],128,t['S4'],t['Z4'],size_n=12288);silu_and_mul_out(y[:,:6144],y[:,6144:],co);return co,y
   operations={'r4_W8A8_two_dot_SiLU':control,'native_W4_plus_same_SiLU':candidate};graphs={};outputs={};keep=[]
   for name,fn in operations.items():
    for _ in range(3):fn()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,pool=pool):result=fn()
    graphs[name]=graph;outputs[name]=result;keep.append(result)
   for phase in range(3):
    with np.load(root/f'X{phase}.npz') as f:a=f['X'].copy()
    x.copy_(torch.from_numpy(a).view(torch.bfloat16));torch.cuda.synchronize();x_version=int(x._version)
    for name,fn in operations.items():
     e=fn();torch.cuda.synchronize();eager=words(e[0] if isinstance(e,tuple) else e);repeat=fn();torch.cuda.synchronize();assert np.array_equal(eager,words(repeat[0] if isinstance(repeat,tuple) else repeat))
     target=outputs[name][0] if isinstance(outputs[name],tuple) else outputs[name];target.fill_(float('nan'));graphs[name].replay();torch.cuda.synchronize();actual=words(target);assert np.array_equal(actual,eager);assert int(x._version)==x_version and np.array_equal(words(x),a)
     path=out/f'L{layer}_{name}_p{phase}.npz';data=dict(output=actual)
     if isinstance(outputs[name],tuple):data['projection']=words(outputs[name][1])
     np.savez(path,**data);state['numeric'].append(dict(layer=layer,phase=phase,variant=name,file=str(path),sha256=sha(path),eager_graph_and_repeat_exact=True));save()
   with np.load(root/'X0.npz') as f:x.copy_(torch.from_numpy(f['X'].copy()).view(torch.bfloat16))
   timed_x_version=int(x._version)
   for mode in ('eager','graph'):
    for rid in range(7):
     order=list(operations);order=order[::-1] if rid%2 else order;times={}
     for name in order:
      fn=graphs[name].replay if mode=='graph' else operations[name];start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
      for _ in range(30):fn()
      end.record();end.synchronize();times[name]=start.elapsed_time(end)/30
     row=dict(layer=layer,mode=mode,round=rid,excluded=rid==0,order=order,ms=times);state['rounds'].append(row);save();print(json.dumps(row),flush=True)
   with np.load(root/'X0.npz') as f:assert int(x._version)==timed_x_version and np.array_equal(words(x),f['X'])
   for k,(version,a) in immutable.items():
    assert int(t[k]._version)==version;actual=words(t[k]) if t[k].dtype==torch.bfloat16 else t[k].cpu().numpy();assert np.array_equal(a,actual),k
   del graphs,outputs,keep,t,operations,x,q,sa,o,co,dummy
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state()) and all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items())
  state.update(complete=True,phase='complete',all_input_full_bytes_versions_preserved=True,RNG_unchanged=True);save()
 except BaseException as e:state.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
 finally:(out/'hardware_after_in_process.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
