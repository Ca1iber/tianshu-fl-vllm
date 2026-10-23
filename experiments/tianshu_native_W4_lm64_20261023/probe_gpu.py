# SPDX-License-Identifier: Apache-2.0
import os,sys,signal,time,subprocess,traceback,runpy,importlib.util
from common import *
def main():
 signal.alarm(900);assert time.time()<1792771080-1200;assert not GPU.exists();GPU.mkdir();gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items()) and sha(MODEL)==gate['checkpoint_sha256'];hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(GPU/'hardware_before.txt').write_text(hw)
 state=dict(complete=False,phase='initializing',pid=os.getpid(),numeric=[],rounds=[],CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),checkpoint_sha256_before=sha(MODEL),model_quality_tested=False,official_benchmark=False,adopted=False,baseline_core_preallocated_buffers=True,LM_actual_r3_runs_outside_model_CUDA_Graph=True)
 def save():(GPU/'result.json').write_text(json.dumps(state,indent=2)+'\n')
 save()
 try:
  profile=json.loads(PROFILE.read_text());env={k:v.replace(profile['old_output_root'],str(GPU)) for k,v in profile['environment'].items()};os.environ.update(env);sys.path[:0]=env['PYTHONPATH'].split(':');runpy.run_path(str(Path(env['PYTHONPATH'])/'sitecustomize.py'))
  import torch,vllm_fl,flag_gems
  torch.set_num_threads(4);path=EXP/'tianshu_dense_w8a8_20261020/kernel.py';assert sha(path)=='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152';spec=importlib.util.spec_from_file_location('_W4_lm_current',path);dense=importlib.util.module_from_spec(spec);sys.modules[spec.name]=dense;spec.loader.exec_module(dense);immutable=[];guards=[]
  def array(name):return np.load(CPU/f'{name}.npy',allow_pickle=False)
  def tensor(a,name,read_only=True):
   a=np.ascontiguousarray(a);t=torch.from_numpy(a.copy());t=t.view(torch.bfloat16) if a.dtype==np.uint16 else t;raw=torch.full((a.nbytes+8192,),113,dtype=torch.uint8,device='cuda');v=raw[4096:-4096].view(t.dtype).view(t.shape);v.copy_(t);guards.append(raw)
   if read_only:immutable.append((name,v,ahash(a),int(v._version)))
   return v
  tensors={name:tensor(array(name),name) for name in ('W8','S8','WA','WD','S4','ZA','ZD')};x=tensor(array('X0'),'X');q=tensor(np.zeros((M,K),np.int8),'q',False);asc=tensor(np.zeros(M,np.float32),'as',False);out=tensor(np.zeros((M,N),np.uint16),'out',False);dummy=torch.empty(1,device='cuda',dtype=torch.int32);config=dict(bm=64,bn=128,bk=128,warps=4,stages=1)
  def control():dense.quantize(x,q,asc);dense.mm(q,tensors['W8'],asc,tensors['S8'],out,dummy,config);return out
  def awq():return torch.ops.aten._weight_int4pack_mm_awq.default(x,tensors['WA'],128,tensors['S4'],tensors['ZA'])
  def dots():return torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,tensors['WD'],128,tensors['S4'],tensors['ZD'],size_n=N)
  compiled=dense.mm(q,tensors['W8'],asc,tensors['S8'],out,dummy,config);state['baseline_compiled']=dict(binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared,configuration=config);assert compiled.n_regs==93 and compiled.n_spills==0 and compiled.metadata.shared==24576
  operations=dict(zip(VARIANTS,[control,awq,dots]));graphs={};outputs={};state['phase']='warming';save()
  for name,fn in operations.items():
   for _ in range(2):fn()
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):o=fn()
   graphs[name]=g;outputs[name]=o
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();state['phase']='numeric';save()
  for phase in range(3):
   xarr=array(f'X{phase}');x.copy_(torch.from_numpy(xarr).view(torch.bfloat16))
   for name,fn in operations.items():
    e=fn();torch.cuda.synchronize();eager=words(e);outputs[name].fill_(float('nan'))
    if name=='W8A8_current':q.fill_(-128);asc.fill_(float('nan'))
    graphs[name].replay();torch.cuda.synchronize();actual=words(outputs[name]);assert np.array_equal(actual,eager) and np.isfinite(f32(actual)).all();p=GPU/f'{name}_phase{phase}.npy';np.save(p,actual,allow_pickle=False);r=dict(variant=name,phase=phase,file=str(p),sha256=sha(p),full_graph_poison_eager_exact=True,all_output_finite=True)
    if name=='W8A8_current':
     p=GPU/f'A8_phase{phase}.npz';np.savez(p,Q=q.cpu().numpy(),S=asc.cpu().numpy());r.update(A8_file=str(p),A8_sha256=sha(p))
    state['numeric'].append(r);save()
   assert ahash(words(x))==ahash(xarr)
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());state['RNG_unchanged']=True;xarr=array('X0');x.copy_(torch.from_numpy(xarr).view(torch.bfloat16))
  for i,e in enumerate(immutable):
   if e[0]=='X':immutable[i]=(e[0],e[1],ahash(xarr),int(x._version))
  state['phase']='timing';save()
  for mode in ('eager','graph'):
   for rid in range(7):
    order=VARIANTS[rid%3:]+VARIANTS[:rid%3];order.reverse() if rid%2 else None;times={}
    for name in order:
     fn=graphs[name].replay if mode=='graph' else operations[name];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
     for _ in range(20):fn()
     b.record();b.synchronize();times[name]=a.elapsed_time(b)/20
    r=dict(mode=mode,round=rid,excluded=rid==0,replays=20,order=order,milliseconds=times);state['rounds'].append(r);save();print(json.dumps(r),flush=True)
  state['phase']='checking_immutability';save()
  for name,t,h,version in immutable:assert t._version==version and ahash(t.contiguous().view(torch.uint8).cpu().numpy())==h,name
  for raw in guards:assert bool((raw[:4096]==113).all() and (raw[-4096:]==113).all())
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_after']==state['checkpoint_sha256_before'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items());state.update(complete=True,phase='complete',checkpoint_sources_and_inputs_unchanged=True,all_guards_versions_full_bytes_preserved=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
 finally:(GPU/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
