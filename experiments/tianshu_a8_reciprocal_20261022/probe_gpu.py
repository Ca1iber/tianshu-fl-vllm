# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from domain_cpu import peaks,values_for_peaks,clamp_words,f32
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args();remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>1800;signal.alarm(min(1800,remain))
 root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(root/'hardware_before.txt').write_text(hw)
 sources=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text())['sources'];assert all(sha(p)==v for p,v in sources.items())
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),sources=sources,domain={},cases=[],rounds=[],model_executed=False,adopted=False)
 def save():
  p=root/'result.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
 save()
 try:
  import torch,vllm_fl,triton
  torch.set_num_threads(4);old=load('_recip_old',ROOT/'tianshu_dense_w8a8_20261020/kernel.py');new=load('_recip_new',BASE/'kernel.py');state['target']=str(triton.runtime.driver.active.get_current_target())
  def guarded(shape,dtype,value):
   count=int(np.prod(shape));store=torch.full((count+512,),value,dtype=dtype,device='cuda');return [store,store[256:count+256].view(shape),value]
  def guards(items):
   for store,_,v in items:assert bool((store[:256]==v).all() and (store[-256:]==v).all())
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def cpu_bf(w):return torch.from_numpy(np.ascontiguousarray(w)).view(torch.bfloat16)
  def metadata(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared)
  def variants(a,buffers):
   m,k=a.shape;result={}
   for name,method in [('old',None),('half_div',1),('half_fma',2)]:
    qbuf=guarded((m,k),torch.int8,17);sbuf=guarded((m,),torch.float32,17.);buffers.extend([qbuf,sbuf]);q,s=qbuf[1],sbuf[1]
    def operation(method=method,q=q,s=s):return old.quantize(a,q,s) if method is None else new.quantize(a,q,s,method)
    compiled=operation();torch.cuda.synchronize();result[name]=dict(q=q,s=s,operation=operation,compiled=metadata(compiled))
   return result
  state['phase']='domain';save();ps=peaks();tiny=clamp_words();extra=np.zeros(((len(tiny)+4095)//4096,4096),np.uint16);extra.ravel()[:len(tiny)]=tiny;domain=np.concatenate([values_for_peaks(ps),extra]);np.save(root/'domain_A_words.npy',domain);abuf=guarded(domain.shape,torch.bfloat16,17.);a=abuf[1];a.copy_(cpu_bf(domain));buffers=[abuf];vs=variants(a,buffers);cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
  for name,v in vs.items():
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):v['operation']()
   graph.replay();torch.cuda.synchronize();q=v['q'].cpu().numpy();s=v['s'].cpu().numpy();np.save(root/f'domain_{name}_q.npy',q);np.save(root/f'domain_{name}_scale.npy',s)
   if name=='old':referenceq=q.copy();references=s.copy()
   else:assert np.array_equal(q,referenceq) and np.array_equal(s.view(np.uint32),references.view(np.uint32))
   state['domain'][name]=dict(compiled=v['compiled'],q_sha256=sha(root/f'domain_{name}_q.npy'),scale_sha256=sha(root/f'domain_{name}_scale.npy'),graph_replayed=True)
  guards(buffers);assert np.array_equal(words(a),domain) and torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng);state['domain'].update(M=domain.shape[0],K=4096,peaks=len(ps),clamp_words=len(tiny),input_sha256=sha(root/'domain_A_words.npy'),guards=True,input_unchanged=True,RNG_unchanged=True);save();print(json.dumps(dict(domain='complete',elements=int(domain.size))),flush=True)
  del vs,v,graph,buffers,abuf,a,domain,extra,q,s,referenceq,references;torch.cuda.empty_cache()
  for m,k,strided in [(8,2048,False),(64,2048,False),(64,6144,False),(17,259,True),(512,2048,False),(4096,2048,False)]:
   rng=np.random.default_rng(202610220000+m+k);ac=rng.standard_normal((m,k),dtype=np.float32);u=ac.view(np.uint32);basewords=((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16);basewords[0]=0;peak_word=np.uint16(12847+30*128);basewords[1]=0;basewords[1,:3]=[peak_word,peak_word-128,(peak_word-128)|0x8000]
   abuf=guarded((m,k+16 if strided else k),torch.bfloat16,17.);a=abuf[1][:,:k];a.copy_(cpu_bf(basewords));buffers=[abuf];vs=variants(a,buffers);case=dict(M=m,K=k,strided=strided,phases=[],compiled={name:v['compiled'] for name,v in vs.items()});state['cases'].append(case);state['phase']=f'case_{m}_{k}';save()
   for name,v in vs.items():
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     for _ in range(32):v['operation']()
    v['graph']=graph
   for phase in range(3):
    factor=1. if phase==0 else -.5 if phase==1 else 2.;actual=f32(basewords)*np.float32(factor);actualwords=(actual.view(np.uint32)>>16).astype(np.uint16);a.copy_(cpu_bf(actualwords));cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();saved=dict(A_words=actualwords)
    for name,v in vs.items():
     v['graph'].replay();torch.cuda.synchronize();saved[name+'_q']=v['q'].cpu().numpy().copy();saved[name+'_scale']=v['s'].cpu().numpy().copy()
     if name!='old':assert np.array_equal(saved[name+'_q'],saved['old_q']) and np.array_equal(saved[name+'_scale'].view(np.uint32),saved['old_scale'].view(np.uint32))
    assert np.array_equal(words(a),actualwords) and torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng);guards(buffers)
    if strided:assert bool((abuf[1][:,k:]==17).all())
    file=f'M{m}_K{k}_phase{phase}.npz';np.savez(root/file,**saved);case['phases'].append(dict(phase=phase,artifact=file,sha256=sha(root/file),full_q_scale_equal=True,guards=True,RNG_unchanged=True,input_unchanged=True));save()
   for roundid in range(7):
    names=['old','half_div','half_fma'];names=names[roundid%3:]+names[:roundid%3];names=names[::-1] if roundid%2 else names;times={}
    for name in names:
     begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
     for _ in range(10):vs[name]['graph'].replay()
     end.record();end.synchronize();times[name]=begin.elapsed_time(end)/320
    state['rounds'].append(dict(M=m,K=k,round=roundid,order=names,nodes_per_graph=32,replays=10,milliseconds_per_quantization=times))
   guards(buffers);case['complete']=True;save();print(json.dumps(dict(M=m,K=k,complete=True)),flush=True);del vs,v,graph,a,abuf,buffers,saved;torch.cuda.empty_cache()
  assert all(sha(p)==v for p,v in sources.items());state.update(complete=True,phase='complete');save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
