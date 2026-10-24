# SPDX-License-Identifier: Apache-2.0
import os,sys,signal,time,subprocess,inspect,runpy,traceback
from datetime import datetime,timezone
from common import *

def main():
 remaining=int((datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds());assert remaining>600;signal.alarm(min(1800,remaining));assert not GPU.exists();GPU.mkdir()
 gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items());assert sha(MODEL)==gate['checkpoint_sha256']
 assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
 state=dict(phase='initializing',complete=False,pid=os.getpid(),sources=gate['sources'],CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),numeric=[],rounds=[],model_quality_tested=False,official_benchmark=False,adopted=False)
 def save():(GPU/'result.json').write_text(json.dumps(state,indent=2)+'\n')
 def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
 save()
 try:
  p=json.loads((CPU/'profile.json').read_text());env=p['environment']|p['additional_environment'];os.environ.update(env);sys.path[:0]=env['PYTHONPATH'].split(':');runpy.run_path(str(Path(env['PYTHONPATH'])/'sitecustomize.py'))
  import importlib,torch,vllm_fl,flag_gems
  from kernel import run
  torch.set_num_threads(4);flag_gems.enable(record=False,once=True);module=importlib.import_module('_iluvatar.ops.mm');assert sha(module.__file__)=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090';launch=Path(inspect.getsourcefile(module._launch_mm));assert launch.name=='sitecustomize.py' and 'tianshu_prefill_gemm_tiles_20261014' in str(launch);state.update(environment=env,actual_BF16_vendor_source=str(Path(module.__file__).resolve()),actual_BF16_vendor_sha256=sha(module.__file__),actual_BF16_launch_source=str(launch),actual_BF16_launch_sha256=sha(launch));save()
  immutable=[];guards=[]
  def tensor(a,name):
   a=np.ascontiguousarray(a);t=torch.from_numpy(a.copy());t=t.view(torch.bfloat16) if a.dtype==np.uint16 else t;raw=torch.full((a.nbytes+8192,),113,dtype=torch.uint8,device='cuda');view=raw[4096:-4096].view(t.dtype).view(t.shape);view.copy_(t);guards.append(raw);immutable.append([name,view,ahash(a),int(view._version)]);return view
  banks=[]
  for e in gate['banks']:
   l=e['layer'];z=np.load(e['file'],allow_pickle=False);banks.append(dict(layer=l,x=tensor(hwords(0,l),f'X{l}'),bf=tensor(np.asarray(weight(l)),f'BF{l}'),w=tensor(z['W'],f'W{l}'),s=tensor(z['S'],f'S{l}'),z=tensor(z['Z'],f'Z{l}')))
  cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();keep=[]
  def phase_x(phase):
   for b in banks:
    a=hwords(phase,b['layer']);b['x'].copy_(torch.from_numpy(a).view(torch.bfloat16));entry=next(v for v in immutable if v[0]==f"X{b['layer']}");entry[2:]=[ahash(a),int(b['x']._version)]
  for m in MS:
   operations={VARIANTS[0]:lambda:[module.mm(b['x'][:m],b['bf'].T) for b in banks],VARIANTS[1]:lambda:[run(b['x'][:m],b['w'],b['s'],b['z']) for b in banks]};graphs={};outputs={};state['phase']=f'M{m}_warm';save()
   for name,fn in operations.items():
    for _ in range(2):fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    graphs[name]=g;outputs[name]=out
   keep.append((graphs,outputs));state['phase']=f'M{m}_numeric';save()
   for phase in PHASES:
    phase_x(phase)
    for name,fn in operations.items():
     eager=fn();torch.cuda.synchronize();first=[words(t) for t in eager];ep=GPU/f'M{m}_{name}_phase{phase}_eager.npz';np.savez(ep,**{f'L{i}':a for i,a in enumerate(first)});repeated=[]
     for repeat in (1,2):
      out=fn();torch.cuda.synchronize();arrays=[words(t) for t in out];exact=all(np.array_equal(a,b) for a,b in zip(first,arrays));row=dict(repeat=repeat,all_42_word_exact=bool(exact))
      if not exact:
       rp=GPU/f'M{m}_{name}_phase{phase}_repeat{repeat}.npz';np.savez(rp,**{f'L{i}':a for i,a in enumerate(arrays)});row.update(file=str(rp),sha256=sha(rp))
      repeated.append(row)
     for t in outputs[name]:t.fill_(float('nan'))
     graphs[name].replay();torch.cuda.synchronize();arrays=[words(t) for t in outputs[name]];path=GPU/f'M{m}_{name}_phase{phase}_graph.npz';np.savez(path,**{f'L{i}':a for i,a in enumerate(arrays)});state['numeric'].append(dict(M=m,phase=phase,variant=name,file=str(path),sha256=sha(path),eager_file=str(ep),eager_sha256=sha(ep),repeats=repeated,all_42_poison_graph_eager_exact=all(np.array_equal(a,b) for a,b in zip(first,arrays)),all_42_finite=all(np.isfinite(f32(a)).all() for a in arrays)));save()
   phase_x(0);state['phase']=f'M{m}_timing';save()
   for mode in ('graph','eager'):
    for rid in range(7):
     order=list(operations)
     if rid%2:order.reverse()
     times={}
     for name in order:
      fn=graphs[name].replay if mode=='graph' else operations[name];start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
      for _ in range(10):fn()
      end.record();end.synchronize();times[name]=start.elapsed_time(end)/10
     row=dict(M=m,mode=mode,round=rid,excluded=rid==0,replays=10,actual_distinct_banks=42,order=order,milliseconds_42banks=times);state['rounds'].append(row);save();print(json.dumps(row),flush=True)
  state['phase']='checking';save()
  for name,t,h,v in immutable:assert t._version==v and ahash(t.contiguous().view(torch.uint8).cpu().numpy())==h,name
  for raw in guards:assert bool((raw[:4096]==113).all() and (raw[-4096:]==113).all())
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items()) and sha(MODEL)==gate['checkpoint_sha256'];state.update(phase='complete',complete=True,RNG_unchanged=True,full_input_and_weight_bytes_versions_guards_unchanged=True,checkpoint_unchanged=True);save()
 except BaseException as e:state.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
 finally:(GPU/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
