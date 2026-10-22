# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args();remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>1800;signal.alarm(min(remain,1800))
 root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(root/'hardware_before.txt').write_text(hw)
 sources=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text())['sources'];assert all(sha(path)==digest for path,digest in sources.items())
 for path,digest in sources.items():
  snapshot=root/'source_snapshot'/Path(path).relative_to('/');snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(path).read_bytes());assert sha(snapshot)==digest
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,cases=[],rounds=[],model_executed=False,official_benchmark=False,adopted=False,checkpoint_sha256_before=sha(MODEL))
 def save():
  tmp=root/'result.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'result.json')
 save()
 try:
  import torch,vllm_fl,triton
  from safetensors import safe_open
  torch.set_num_threads(4);quant=load('_prefill_native_quant',ROOT/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py');current=load('_prefill_native_current',ROOT/'tianshu_nn_silu_large_tiles_v2_20261021/kernel.py');new=load('_prefill_native_epi',BASE/'kernel.py')
  selected=json.loads((ROOT/'tianshu_m_range_prefill_service_v2_20261021/selected_plan.json').read_text());component=json.loads(Path(selected['component_gate']).read_text());assert component['complete'] and sha(selected['component_gate'])==selected['component_gate_sha256'];state['target']=str(triton.runtime.driver.active.get_current_target());state['scalar_cache_key']=current._native_silu_scalar.cache_key;assert new._native_silu_scalar.cache_key==state['scalar_cache_key']
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint:original=torch.cat([checkpoint.get_tensor(f'model.layers.0.mlp.{role}_proj.weight') for role in ('gate','up')],0).contiguous()
  values=original.float();scales=values.abs().amax(1).clamp_min(1.e-8)/127.;codes=(values/scales[:,None]).round().clamp(-127,127).to(torch.int8);n,k=original.shape;i=n//2;assert (n,k)==(12288,2048)
  def hash_tensor(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  state['weight_signature']=dict(original_BF16_sha256=hash_tensor(original),packed_sha256=hash_tensor(codes),scales_sha256=hash_tensor(scales));expected=next(r for r in json.loads((ROOT/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text()) if r['prefix']=='model.layers.0.mlp.gate_up_proj');assert all(expected[key]==value for key,value in state['weight_signature'].items())
  wn,wt,ws=codes.T.contiguous().cuda(),codes.cuda(),scales.cuda()
  def guarded(shape,dtype,value):
   count=int(np.prod(shape));storage=torch.full((count+512,),value,dtype=dtype,device='cuda');return storage,storage[256:count+256].view(shape),value
  def guards(buffers):
   for storage,_,value in buffers:assert bool((storage[:256]==value).all() and (storage[-256:]==value).all())
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def array_hash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
  def metadata(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared)
  for m in (128,512,4096):
   config=selected['configuration_by_M'][str(m)];sample=np.random.default_rng(202610221200+m).standard_normal((m,k),dtype=np.float32);u=sample.view(np.uint32);base_words=((u+32767+((u>>16)&1))>>16).astype(np.uint16);base_words[0]=0;ac=torch.from_numpy(base_words).view(torch.bfloat16);buffers=[];arrays={}
   for field,shape,dtype,value in [('a',(m,k),torch.bfloat16,17.),('q',(m,k),torch.int8,17),('as',(m,),torch.float32,17.),('old_out',(m,i),torch.bfloat16,17.),('old_int',(m,n),torch.int32,-777),('old_proj',(m,n),torch.bfloat16,17.),('integer',(m,n),torch.int32,-777),('projected',(m,n),torch.bfloat16,17.)]:
    buf=guarded(shape,dtype,value);buffers.append(buf);arrays[field]=buf[1]
   a,q,asc=arrays['a'],arrays['q'],arrays['as'];integer=arrays['integer'];a.copy_(ac);dummy=torch.empty(1,dtype=torch.int32,device='cuda')
   def baseline():
    quant.quantize(a,q,asc);return current.fused(q,wn,asc,ws,arrays['old_out'],dummy,arrays['old_out'],config)
   binary=baseline();torch.cuda.synchronize();assert metadata(binary)['binary_sha256']==component['actual_timed_candidate_kernels'][str(m)]['binary_sha256'];graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):baseline()
   old_graph=graph;variants={}
   for layout in ('NN','NT'):
    weight=wn if layout=='NN' else wt.T
    for rows in (1,4):
     name=f'{layout}_R{rows}';buf=guarded((m,i),torch.bfloat16,17.);buffers.append(buf);out=buf[1]
     def operation(weight=weight,rows=rows,out=out):
      quant.quantize(a,q,asc);torch._int_mm(q,weight,out=integer);return new.epilogue(integer,asc,ws,out,arrays['projected'],rows)
     compiled=operation();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):operation()
     variants[name]=dict(graph=graph,out=out,layout=layout,rows=rows,compiled=metadata(compiled))
   case=dict(M=m,K=k,N=n,current_config=config,current_compiled=metadata(binary),native_workspace_bytes=m*n*4,variants={name:{key:value for key,value in v.items() if key not in ('graph','out')} for name,v in variants.items()},phases=[]);state['cases'].append(case);state['phase']=f'M{m}';save();r=np.unique(np.linspace(0,m-1,8).astype(np.int64));c=np.unique(np.linspace(0,i-1,32).astype(np.int64));both=np.concatenate([c,c+i])
   for phase in range(3):
    actual=ac if phase==0 else ac*(.5 if phase==1 else -.5);sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.;a.copy_(actual);wn.copy_((codes*sign).T.contiguous());wt.copy_(codes*sign);ws.copy_(scales*factor);before_a=a.clone();before_wn=wn.clone();before_wt=wt.clone();before_ws=ws.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
    old_graph.replay();torch.cuda.synchronize();old_words=words(arrays['old_out']);current.fused(q,wn,asc,ws,arrays['old_out'],arrays['old_int'],arrays['old_proj'],config,diagnostics=True);torch.cuda.synchronize();assert np.array_equal(old_words,words(arrays['old_out']));proj=words(arrays['old_proj']);projection_hash=array_hash(proj);output_hash=array_hash(old_words)
    artifact=root/f'M{m}_phase{phase}.npz';np.savez(artifact,A_words=words(a),Aq=q.cpu().numpy(),As=asc.cpu().numpy(),WS=ws.cpu().numpy(),integer=arrays['old_int'].cpu().numpy(),rows=r,columns=c,projected_selected_words=proj[np.ix_(r,both)],current_output_selected_words=old_words[np.ix_(r,c)])
    entries=[]
    for name,v in variants.items():
     v['graph'].replay();torch.cuda.synchronize();timed=words(v['out']);assert np.array_equal(timed,old_words) and torch.equal(integer,arrays['old_int']);actual_output_hash=array_hash(timed)
     diagnostic=new.epilogue(integer,asc,ws,v['out'],arrays['projected'],v['rows'],True);torch.cuda.synchronize();assert torch.equal(arrays['projected'].view(torch.int16),arrays['old_proj'].view(torch.int16)) and np.array_equal(words(v['out']),timed);actual_projection_hash=array_hash(words(arrays['projected']));assert actual_projection_hash==projection_hash and actual_output_hash==output_hash
     entries.append(dict(variant=name,full_INT32_exact=True,full_projected_BF16_sha256=actual_projection_hash,full_output_BF16_sha256=actual_output_hash,diagnostic=metadata(diagnostic)))
    assert torch.equal(a,before_a) and torch.equal(wn,before_wn) and torch.equal(wt,before_wt) and torch.equal(ws,before_ws) and torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng);guards(buffers)
    case['phases'].append(dict(phase=phase,weight_sign=sign,weight_scale_factor=factor,artifact=artifact.name,sha256=sha(artifact),current_projection_BF16_sha256=projection_hash,current_output_BF16_sha256=output_hash,entries=entries,guards_inputs_weights_RNG=True));save()
   graphs={'current':old_graph,**{name:v['graph'] for name,v in variants.items()}}
   for roundid in range(7):
    names=list(graphs);shift=(2*roundid)%len(names);names=names[shift:]+names[:shift]
    if roundid%2:names.reverse()
    times={}
    for name in names:
     begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
     for _ in range(20):graphs[name].replay()
     end.record();end.synchronize();times[name]=begin.elapsed_time(end)/20
    state['rounds'].append(dict(M=m,round=roundid,order=names,replays=20,milliseconds=times))
   guards(buffers);case['complete']=True;save();print(json.dumps(dict(M=m,complete=True)),flush=True)
   del arrays,buffers,buf,a,q,asc,integer,out,variants,v,graphs,graph,old_graph,before_a,before_wn,before_wt,before_ws,proj,old_words,timed;torch.cuda.empty_cache()
  state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_before']==state['checkpoint_sha256_after'] and all(sha(path)==digest for path,digest in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
