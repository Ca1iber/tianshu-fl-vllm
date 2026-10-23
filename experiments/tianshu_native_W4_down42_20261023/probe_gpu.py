# SPDX-License-Identifier: Apache-2.0
import os,sys,signal,subprocess,time,inspect,runpy,importlib,traceback
from common import *
def main():
 signal.alarm(900);assert time.time()<1792771080-900
 assert not GPU.exists();GPU.mkdir();gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items()) and sha(MODEL)==gate['checkpoint_sha256']
 hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(GPU/'hardware_before.txt').write_text(hw)
 state=dict(complete=False,phase='initializing',pid=os.getpid(),numeric=[],rounds=[],quantization_errors=[],checkpoint_sha256_before=sha(MODEL),sources=gate['sources'],CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),model_quality_tested=False,official_benchmark=False,adopted=False)
 def save():(GPU/'result.json').write_text(json.dumps(state,indent=2)+'\n')
 save()
 try:
  profile=json.loads(PROFILE.read_text());env={k:v.replace(profile['old_output_root'],str(GPU)) for k,v in profile['environment'].items()};os.environ.update(env);sys.path[:0]=env['PYTHONPATH'].split(':');runpy.run_path(str(Path(env['PYTHONPATH'])/'sitecustomize.py'))
  import torch,vllm_fl,flag_gems
  torch.set_num_threads(4);module=importlib.import_module('_iluvatar.ops.mm');state['baseline']=dict(module=module.__name__,source=module.__file__,source_sha256=sha(module.__file__),launch_source=inspect.getsourcefile(module._launch_mm),launch_name=module._launch_mm.__qualname__);assert state['baseline']['source_sha256']=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090';assert state['baseline']['launch_source'].endswith('tianshu_prefill_gemm_tiles_20261014/hooks/sitecustomize.py');state['environment']=env;save()
  immutable=[];guarded=[]
  def tensor(arr,name):
   arr=np.ascontiguousarray(arr);t=torch.from_numpy(arr.copy());t=t.view(torch.bfloat16) if arr.dtype==np.uint16 else t;raw=torch.full((arr.nbytes+8192,),113,dtype=torch.uint8,device='cuda');view=raw[4096:-4096].view(t.dtype).view(t.shape);view.copy_(t);guarded.append(raw);immutable.append((name,view,ahash(arr),int(view._version)));return view
  banks=[]
  for entry in gate['banks']:
   layer=entry['layer'];data=np.load(entry['file'],allow_pickle=False);banks.append(dict(layer=layer,x=tensor(hwords(0,layer),f'X{layer}'),bf16=tensor(np.asarray(weight(layer)),f'BF16{layer}'),w=tensor(data['W'],f'W4{layer}'),s=tensor(data['S'],f'S{layer}'),z=tensor(data['Z'],f'Z{layer}')))
  def bf16_run():return [module.mm(b['x'],b['bf16'].T) for b in banks]
  def w4_run():return [torch.ops.aten._weight_int4pack_mm_awq.default(b['x'],b['w'],128,b['s'],b['z']) for b in banks]
  operations={'BF16_qualified':bf16_run,'W4_AWQ_RTN':w4_run};outputs={};graphs={};state['phase']='warming';save()
  for name,fn in operations.items():
   for _ in range(2):fn()
   torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=fn()
   graphs[name]=graph;outputs[name]=out
  state['phase']='numeric';save();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  for phase in PHASES:
   for b in banks:
    arr=hwords(phase,b['layer']);b['x'].copy_(torch.from_numpy(arr).view(torch.bfloat16));entry=next(v for v in immutable if v[0]==f"X{b['layer']}");immutable[immutable.index(entry)]=(entry[0],entry[1],ahash(arr),int(b['x']._version))
   phase_outputs={}
   for name,fn in operations.items():
    eager=fn();torch.cuda.synchronize();expected=[words(t) for t in eager]
    for t in outputs[name]:t.fill_(float('nan'))
    graphs[name].replay();torch.cuda.synchronize();actual=[words(t) for t in outputs[name]];assert all(np.array_equal(a,b) for a,b in zip(actual,expected));assert all(np.isfinite(f32(a)).all() for a in actual);path=GPU/f'{name}_phase{phase}.npz';np.savez(path,**{f'L{layer}':a for layer,a in enumerate(actual)});phase_outputs[name]=actual
    state['numeric'].append(dict(phase=phase,variant=name,file=str(path),sha256=sha(path),all_42_graph_poison_eager_exact=True,all_outputs_finite=True));save()
   for layer in range(42):
    original=f32(phase_outputs['BF16_qualified'][layer]).astype(np.float64);quantized=f32(phase_outputs['W4_AWQ_RTN'][layer]).astype(np.float64);state['quantization_errors'].append(dict(phase=phase,layer=layer,relative_L2=float(np.linalg.norm(quantized-original)/max(np.linalg.norm(original),1e-30)),max_abs=float(abs(quantized-original).max())))
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());state['RNG_unchanged']=True
  # Restore phase0; timing uses distinct real banks and independently allocated X per layer.
  for b in banks:
   arr=hwords(0,b['layer']);b['x'].copy_(torch.from_numpy(arr).view(torch.bfloat16))
   for i,e in enumerate(immutable):
    if e[0]==f"X{b['layer']}":immutable[i]=(e[0],e[1],ahash(arr),int(b['x']._version));break
  state['phase']='timing';save()
  for mode in ('graph','eager'):
   for rid in range(7):
    order=list(operations);order.reverse() if rid%2 else None;times={}
    for name in order:
     fn=graphs[name].replay if mode=='graph' else operations[name];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
     for _ in range(10):fn()
     b.record();b.synchronize();times[name]=a.elapsed_time(b)/10
    record=dict(mode=mode,round=rid,excluded=rid==0,replays=10,banks_per_replay=42,order=order,milliseconds_42banks=times);state['rounds'].append(record);save();print(json.dumps(record),flush=True)
  state['phase']='checking_immutability';save()
  for name,t,expected,version in immutable:assert t._version==version and ahash(t.contiguous().view(torch.uint8).cpu().numpy())==expected,name
  for raw in guarded:assert bool((raw[:4096]==113).all() and (raw[-4096:]==113).all())
  state['all_input_weight_scale_zero_full_bytes_versions_guards_preserved']=True;state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_before']==state['checkpoint_sha256_after'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items());state.update(complete=True,phase='complete',checkpoint_unchanged=True,CPU_references_not_yet_verified=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
 finally:(GPU/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
