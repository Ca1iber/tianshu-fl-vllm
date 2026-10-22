# SPDX-License-Identifier: Apache-2.0
"""Real W8 GateUp: current fused path vs native INT32 GEMM plus exact epilogue."""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
CONFIG=dict(bm=64,bn=64,bk=128,warps=4,stages=1,mode='two_dot')

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
 remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>1800;signal.alarm(min(remain,1800))
 root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
 hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(root/'hardware_before.txt').write_text(hw)
 sources=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text())['sources'];assert all(sha(path)==digest for path,digest in sources.items())
 for path,digest in sources.items():
  snapshot=root/'source_snapshot'/Path(path).relative_to('/');snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(path).read_bytes());assert sha(snapshot)==digest
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,cases=[],rounds=[],model_executed=False,official_benchmark=False,adopted=False,checkpoint_sha256_before=sha(MODEL))
 def save():
  temp=root/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'result.json')
 save()
 try:
  import torch,vllm_fl,triton
  from safetensors import safe_open
  torch.set_num_threads(4);nn=load('_native_gateup_NN',ROOT/'tianshu_int8_weight_nn_20261021/kernel_nn.py');fused=load('_native_gateup_current',ROOT/'tianshu_nn_silu_two_dot_20261021/kernel.py');candidate=load('_native_gateup_epilogue',BASE/'kernel.py')
  state['target']=str(triton.runtime.driver.active.get_current_target());state['scalar_cache_key']=fused._native_silu_scalar.cache_key
  assert candidate._native_silu_scalar.cache_key==state['scalar_cache_key']
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint:
   original=torch.cat([checkpoint.get_tensor(f'model.layers.0.mlp.{role}_proj.weight') for role in ('gate','up')],0).contiguous()
  values=original.float();scale=values.abs().amax(1).clamp_min(1.e-8)/127.;codes=(values/scale[:,None]).round().clamp(-127,127).to(torch.int8);m,n,k=64,*original.shape;i=n//2;assert (n,k,i)==(12288,2048,6144)
  def tensor_sha(tensor):return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  proof=dict(original_BF16_sha256=tensor_sha(original),packed_sha256=tensor_sha(codes),scales_sha256=tensor_sha(scale))
  expected=next(row for row in json.loads((ROOT/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text()) if row['prefix']=='model.layers.0.mlp.gate_up_proj');assert all(expected[key]==value for key,value in proof.items());state['weight_signature']=proof
  state['per_native_INT32_workspace_bytes']=m*n*4;state['per_W8_layout_bytes']=n*k
  weight_nt,weight_nn=codes.cuda(),codes.T.contiguous().cuda();weight_scale=scale.cuda()
  def guarded(shape,dtype,value):
   count=int(np.prod(shape));storage=torch.full((count+512,),value,dtype=dtype,device='cuda');return storage,storage[256:count+256].view(shape),value
  def guards(buffers):
   for storage,_,value in buffers:assert bool((storage[:256]==value).all() and (storage[-256:]==value).all())
  def words(tensor):return tensor.contiguous().view(torch.uint16).cpu().numpy().copy()
  def metadata(compiled):return dict(hash=compiled.hash,binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared)
  qualified=next(row for row in json.loads((ROOT/'tianshu_nn_silu_two_dot_20261021/GPU_GATE_32g.json').read_text())['best_by_shape'] if row['M']==64 and not row['strided']);assert qualified['configuration']==CONFIG
  for strided in (False,True):
   rng=np.random.default_rng(202610221100+int(strided));sample=rng.standard_normal((m,k),dtype=np.float32);u=sample.view(np.uint32);base_words=((u+32767+((u>>16)&1))>>16).astype(np.uint16);base_words[0]=0
   ac=torch.from_numpy(base_words).view(torch.bfloat16);abuf=guarded((m,k+7 if strided else k),torch.bfloat16,17.);a=abuf[1][:,:k];a.copy_(ac);buffers=[abuf];arrays={}
   for field,shape,dtype,value in [('q',(m,k),torch.int8,17),('as',(m,),torch.float32,17.),('old_out',(m,i),torch.bfloat16,17.),('old_int',(m,n),torch.int32,-777),('old_proj',(m,n),torch.bfloat16,17.),('native_int',(m,n),torch.int32,-777),('native_proj',(m,n),torch.bfloat16,17.)]:
    item=guarded(shape,dtype,value);buffers.append(item);arrays[field]=item[1]
   q,ascale=arrays['q'],arrays['as'];integer=arrays['native_int'];variants={}
   def baseline():
    nn.quantize(a,q,ascale);return fused.fused(q,weight_nn,ascale,weight_scale,arrays['old_out'],arrays['old_int'],arrays['old_proj'],CONFIG)
   current=baseline();torch.cuda.synchronize();assert metadata(current)['binary_sha256']==qualified['actual_timed_compiled']['binary_sha256']
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):baseline()
   old_graph=graph
   for layout in ('NN','NT'):
    weight=weight_nn if layout=='NN' else weight_nt.T
    for block in (256,512,1024):
     for warps in (4,8):
      name=f'{layout}_B{block}W{warps}';item=guarded((m,i),torch.bfloat16,17.);buffers.append(item);out=item[1]
      def operation(weight=weight,out=out,block=block,warps=warps):
       nn.quantize(a,q,ascale);torch._int_mm(q,weight,out=integer);return candidate.epilogue(integer,ascale,weight_scale,out,arrays['native_proj'],block,warps)
      compiled=operation();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):operation()
      variants[name]=dict(graph=graph,out=out,block=block,warps=warps,layout=layout,compiled=metadata(compiled))
   case=dict(M=m,K=k,N=n,strided=strided,current_compiled=metadata(current),configuration=CONFIG,variants={name:{key:value for key,value in v.items() if key not in ('graph','out')} for name,v in variants.items()},phases=[]);state['cases'].append(case);state['phase']=f'strided{int(strided)}';save()
   rows=np.unique(np.linspace(0,m-1,8).astype(np.int64));columns=np.unique(np.linspace(0,i-1,32).astype(np.int64));both=np.concatenate([columns,columns+i])
   for phase in range(3):
    actual_a=ac if phase==0 else ac*(.5 if phase==1 else -.5);sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.
    a.copy_(actual_a);weight_nt.copy_(codes*sign);weight_nn.copy_((codes*sign).T.contiguous());weight_scale.copy_(scale*factor);before_a=abuf[1].clone();before_nt=weight_nt.clone();before_nn=weight_nn.clone();before_ws=weight_scale.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
    old_graph.replay();torch.cuda.synchronize();old_words=words(arrays['old_out']);fused.fused(q,weight_nn,ascale,weight_scale,arrays['old_out'],arrays['old_int'],arrays['old_proj'],CONFIG,diagnostics=True);torch.cuda.synchronize();assert np.array_equal(old_words,words(arrays['old_out']))
    entries=[]
    for name,v in variants.items():
     v['graph'].replay();torch.cuda.synchronize();timed_words=words(v['out']);assert np.array_equal(timed_words,old_words) and torch.equal(integer,arrays['old_int'])
     diagnostic=candidate.epilogue(integer,ascale,weight_scale,v['out'],arrays['native_proj'],v['block'],v['warps'],diagnostics=True);torch.cuda.synchronize();assert np.array_equal(timed_words,words(v['out'])) and torch.equal(arrays['native_proj'].view(torch.int16),arrays['old_proj'].view(torch.int16))
     file=root/f'strided{int(strided)}_{name}_phase{phase}.npz';np.savez(file,A_words=words(a),Aq=q.cpu().numpy(),As=ascale.cpu().numpy(),WS=weight_scale.cpu().numpy(),integer=integer.cpu().numpy(),projected_words=words(arrays['native_proj']),output_words=timed_words,current_output_words=old_words,rows=rows,columns=columns)
     entries.append(dict(variant=name,artifact=file.name,sha256=sha(file),full_INT32_BF16_projection_output_exact=True,diagnostic=metadata(diagnostic)))
    assert torch.equal(abuf[1],before_a) and torch.equal(weight_nt,before_nt) and torch.equal(weight_nn,before_nn) and torch.equal(weight_scale,before_ws) and torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng);guards(buffers)
    case['phases'].append(dict(phase=phase,weight_sign=sign,weight_scale_factor=factor,entries=entries,guards_inputs_weights_RNG=True));save()
   graphs={'current':old_graph,**{name:v['graph'] for name,v in variants.items()}}
   for roundid in range(7):
    names=list(graphs);shift=(5*roundid)%len(names);names=names[shift:]+names[:shift]
    if roundid%2:names.reverse()
    times={}
    for name in names:
     begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
     for _ in range(20):graphs[name].replay()
     end.record();end.synchronize();times[name]=begin.elapsed_time(end)/20
    state['rounds'].append(dict(strided=strided,round=roundid,order=names,replays=20,milliseconds=times))
   guards(buffers);case['complete']=True;save();print(json.dumps(dict(strided=strided,complete=True)),flush=True)
   del graphs,variants,v,graph,old_graph,buffers,arrays,q,ascale,integer,a,abuf,before_a,before_nt,before_nn,before_ws,out,item;torch.cuda.empty_cache()
  state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];assert all(sha(path)==digest for path,digest in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
