# SPDX-License-Identifier: Apache-2.0
"""Full-output exact comparison and rotating CUDA Graph geometry timings."""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
CASES=[(8,2048,False),(64,2048,False),(64,6144,False),(17,259,True),(128,2048,False),(512,2048,False),(2048,2048,False),(4096,2048,False)]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def words_to_float(a):return (np.asarray(a,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16_words(a):
 u=np.asarray(a,np.float32).view(np.uint32);return ((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
def fixture(m,k,phase):
 values=np.random.default_rng(202610220000+m+k).standard_normal((m,k),dtype=np.float32);base=bf16_words(values)
 base[0]=0;peak=np.uint16(12847+30*128);base[1]=0;base[1,:3]=[peak,peak-128,(peak-128)|0x8000]
 base[2,:7]=bf16_words(np.array([0,.5,1.5,2.5,-.5,-1.5,-2.5],np.float32));base[3]=np.uint16(1)
 actual=words_to_float(base)
 if phase==1:actual=actual*np.float32(-.5)
 if phase==2:actual=actual[:,::-1].copy()+np.float32(.125)
 result=bf16_words(actual);result[0]=0;return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
 remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>1800;signal.alarm(min(remain,1800))
 root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(root/'hardware_before.txt').write_text(hw)
 sources=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text())['sources'];assert all(sha(p)==v for p,v in sources.items())
 for path,digest in sources.items():
  snapshot=root/'source_snapshot'/Path(path).relative_to('/');snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(path).read_bytes());assert sha(snapshot)==digest
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),sources=sources,cases=[],rounds=[],model_executed=False,adopted=False)
 def save():
  path=root/'result.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)
 save()
 try:
  import torch,vllm_fl,triton
  torch.set_num_threads(4);old=load('_geometry_old',ROOT/'tianshu_dense_w8a8_20261020/kernel.py');new=load('_geometry_new',BASE/'kernel.py');configs=json.loads((BASE/'configs.json').read_text());state['target']=str(triton.runtime.driver.active.get_current_target())
  def guarded(shape,dtype,value):
   count=int(np.prod(shape));storage=torch.full((count+512,),value,dtype=dtype,device='cuda');return storage,storage[256:count+256].view(shape),value
  def guards(items):
   for storage,_,value in items:assert bool((storage[:256]==value).all() and (storage[-256:]==value).all())
  def metadata(compiled):return dict(hash=compiled.hash,binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared)
  for m,k,strided in CASES:
   abuf=guarded((m,k+16 if strided else k),torch.bfloat16,17.);a=abuf[1][:,:k];a.copy_(torch.from_numpy(fixture(m,k,0)).view(torch.bfloat16));buffers=[abuf];variants={}
   for name,rows,warps in [('old',0,0)]+[(f'R{r}W{w}',r,w) for r,w in configs]:
    qbuf=guarded((m,k),torch.int8,17);sbuf=guarded((m,),torch.float32,17.);buffers.extend([qbuf,sbuf]);q,s=qbuf[1],sbuf[1]
    def operation(rows=rows,warps=warps,q=q,s=s):return old.quantize(a,q,s) if rows==0 else new.quantize(a,q,s,rows,warps)
    compiled=operation();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     for _ in range(32):operation()
    variants[name]=dict(q=q,s=s,graph=graph,compiled=metadata(compiled))
   case=dict(M=m,K=k,strided=strided,compiled={name:v['compiled'] for name,v in variants.items()},phases=[]);state['cases'].append(case);state['phase']=f'M{m}_K{k}';save()
   for phase in range(3):
    actual=fixture(m,k,phase);a.copy_(torch.from_numpy(actual).view(torch.bfloat16));cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();saved=dict(A_words=actual)
    for name,v in variants.items():
     v['graph'].replay();torch.cuda.synchronize();saved[name+'_q']=v['q'].cpu().numpy().copy();saved[name+'_scale']=v['s'].cpu().numpy().copy()
     assert name=='old' or (np.array_equal(saved[name+'_q'],saved['old_q']) and np.array_equal(saved[name+'_scale'].view(np.uint32),saved['old_scale'].view(np.uint32)))
    assert np.array_equal(a.contiguous().view(torch.uint16).cpu().numpy(),actual) and torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng);guards(buffers)
    if strided:assert bool((abuf[1][:,k:]==17).all())
    path=root/f'M{m}_K{k}_phase{phase}.npz';np.savez(path,**saved);case['phases'].append(dict(phase=phase,artifact=path.name,sha256=sha(path),all_variants_q_scale_equal=True,guards=True,RNG_unchanged=True,input_unchanged=True));save()
   for roundid in range(7):
    names=list(variants);shift=(7*roundid)%len(names);names=names[shift:]+names[:shift]
    if roundid%2:names.reverse()
    times={}
    for name in names:
     begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
     for _ in range(10):variants[name]['graph'].replay()
     end.record();end.synchronize();times[name]=begin.elapsed_time(end)/320
    state['rounds'].append(dict(M=m,K=k,round=roundid,order=names,nodes_per_graph=32,replays=10,milliseconds_per_quantization=times))
   guards(buffers);case['complete']=True;save();print(json.dumps(dict(M=m,K=k,complete=True)),flush=True);del variants,v,graph,buffers,abuf,a,saved,qbuf,sbuf,q,s;torch.cuda.empty_cache()
  assert all(sha(p)==v for p,v in sources.items());state.update(complete=True,phase='complete');save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
