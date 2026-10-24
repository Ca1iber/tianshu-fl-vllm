# SPDX-License-Identifier: Apache-2.0
import os,sys,signal,time,runpy,inspect,subprocess,traceback
from datetime import datetime,timezone
from common import *
def main():
 remain=int((datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds());assert remain>600;signal.alarm(min(1800,remain));assert not GPU.exists();GPU.mkdir();gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items());assert sha(MODEL)==gate['checkpoint_sha256'];assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
 state=dict(complete=False,phase='initializing',pid=os.getpid(),sources=gate['sources'],CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),numeric=[],rounds=[],model_quality_tested=False,official_benchmark=False,adopted=False)
 def save():(GPU/'result.json').write_text(json.dumps(state,indent=2)+'\n')
 save()
 try:
  p=json.loads((CPU/'profile.json').read_text());env=p['environment']|p['additional_environment'];os.environ.update(env);sys.path[:0]=env['PYTHONPATH'].split(':');runpy.run_path(str(Path(env['PYTHONPATH'])/'sitecustomize.py'))
  import torch,importlib,vllm_fl,flag_gems
  torch.set_num_threads(4);flag_gems.enable(record=False,once=True);vendor=importlib.import_module('_iluvatar.ops.mm');assert sha(vendor.__file__)=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090';launch=Path(inspect.getsourcefile(vendor._launch_mm));assert 'tianshu_prefill_gemm_tiles_20261014' in str(launch);state.update(actual_BF16_source=str(Path(vendor.__file__).resolve()),actual_BF16_sha256=sha(vendor.__file__),actual_BF16_launch_source=str(launch),actual_BF16_launch_sha256=sha(launch),environment=env);save()
  immutable=[];guards=[]
  def tensor(a,name):
   a=np.ascontiguousarray(a);t=torch.from_numpy(a.copy());t=t.view(torch.bfloat16) if a.dtype==np.uint16 else t;raw=torch.full((a.nbytes+8192,),113,dtype=torch.uint8,device='cuda');v=raw[4096:-4096].view(t.dtype).view(t.shape);v.copy_(t);guards.append(raw);immutable.append([name,v,ahash(a),int(v._version)]);return v
  banks=[]
  for item in gate['banks']:
   l,role=item['layer'],item['role'];b=dict(layer=l,role=role,n=item['N'],x=tensor(inputs(l,role,0),f'X{role}{l}'),bf=tensor(original(l,role),f'BF{role}{l}'),variants={})
   for mode in MODES:
    a=np.load(item['variants'][mode]['file'],allow_pickle=False);b['variants'][mode]=[tensor(a[k],f'{role}{l}{mode}{k}') for k in ('W','S','Z')]
   banks.append(b)
  cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();keep=[]
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def phase_x(phase):
   for b in banks:
    a=inputs(b['layer'],b['role'],phase);b['x'].copy_(torch.from_numpy(a).view(torch.bfloat16));entry=next(v for v in immutable if v[0]==f"X{b['role']}{b['layer']}");entry[2:]=[ahash(a),int(b['x']._version)]
  for role in ROLES:
   selected=[b for b in banks if b['role']==role]
   def control():return [vendor.mm(b['x'],b['bf'].T) for b in selected]
   def candidate(mode):return [torch.ops.aten._weight_int4pack_mm_dotsaddz.default(b['x'],*b['variants'][mode][:1],128,*b['variants'][mode][1:],size_n=b['n']) for b in selected]
   operations={VARIANTS[0]:control,VARIANTS[1]:lambda:candidate('sym'),VARIANTS[2]:lambda:candidate('affine')};graphs={};outs={};state['phase']=role+'_warming';save()
   for name,fn in operations.items():
    for _ in range(2):fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    graphs[name]=g;outs[name]=out
   keep.append((graphs,outs))
   for phase in PHASES:
    phase_x(phase);state['phase']=f'{role}_numeric{phase}';save()
    for name,fn in operations.items():
     eager=fn();torch.cuda.synchronize();expected=[words(t) for t in eager];ep=GPU/f'{role}_{name}_phase{phase}_eager.npz';np.savez(ep,**{f'L{i}':a for i,a in enumerate(expected)});repeats=[]
     for repeat in (1,2):
      outputs=fn();torch.cuda.synchronize();actual=[words(t) for t in outputs];exact=all(np.array_equal(a,b) for a,b in zip(expected,actual));row=dict(repeat=repeat,all_42_word_exact=bool(exact))
      if not exact:p=GPU/f'{role}_{name}_phase{phase}_repeat{repeat}.npz';np.savez(p,**{f'L{i}':a for i,a in enumerate(actual)});row.update(file=str(p),sha256=sha(p))
      repeats.append(row)
     for t in outs[name]:t.fill_(float('nan'))
     graphs[name].replay();torch.cuda.synchronize();actual=[words(t) for t in outs[name]];p=GPU/f'{role}_{name}_phase{phase}_graph.npz';np.savez(p,**{f'L{i}':a for i,a in enumerate(actual)});state['numeric'].append(dict(role=role,phase=phase,variant=name,file=str(p),sha256=sha(p),eager_file=str(ep),eager_sha256=sha(ep),repeats=repeats,all_42_poison_graph_eager_exact=all(np.array_equal(a,b) for a,b in zip(expected,actual)),all_42_finite=all(np.isfinite(f32(a)).all() for a in actual)));save()
   phase_x(0);state['phase']=role+'_timing';save()
   for mode in ('graph','eager'):
    for rid in range(7):
     order=list(operations)
     if rid%2:order.reverse()
     times={}
     for name in order:
      fn=graphs[name].replay if mode=='graph' else operations[name];start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
      for _ in range(10):fn()
      end.record();end.synchronize();times[name]=start.elapsed_time(end)/10
     row=dict(role=role,mode=mode,round=rid,excluded=rid==0,replays=10,actual_distinct_banks=42,order=order,milliseconds_42banks=times);state['rounds'].append(row);save();print(json.dumps(row),flush=True)
  state['phase']='checking';save()
  for name,t,h,v in immutable:assert t._version==v and ahash(t.contiguous().view(torch.uint8).cpu().numpy())==h,name
  for raw in guards:assert bool((raw[:4096]==113).all() and (raw[-4096:]==113).all())
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items()) and sha(MODEL)==gate['checkpoint_sha256'];state.update(phase='complete',complete=True,RNG_unchanged=True,full_inputs_weights_versions_guards_unchanged=True,checkpoint_unchanged=True);save()
 except BaseException as e:state.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
 finally:(GPU/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
