# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
 remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>1800;signal.alarm(min(remain,3600))
 root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
 freeze=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());sources=freeze['sources'];assert freeze['complete'] and all(sha(p)==h for p,h in sources.items())
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,cases=[],rounds=[],model_executed=False,official_benchmark=False,adopted=False,checkpoint_sha256_before=sha(MODEL))
 def save():
  temp=root/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'result.json')
 save()
 try:
  import torch,vllm_fl,triton
  from safetensors import safe_open
  torch.set_num_threads(4)
  quant=load('_curve_quant',ROOT/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py');fused=load('_curve_fused',ROOT/'tianshu_nn_silu_large_tiles_v2_20261021/kernel.py');native=load('_curve_native',ROOT/'tianshu_native_int8_prefill_v2_20261022/kernel.py');activation=load('_curve_activation',BASE/'inductor_activation.py')
  selection=json.loads((ROOT/'tianshu_m_range_prefill_service_v2_20261021/selected_plan.json').read_text());component=json.loads(Path(selection['component_gate']).read_text());assert sha(selection['component_gate'])==selection['component_gate_sha256'] and component['complete']
  configurations=selection['configuration_by_M'];projection=json.loads((ROOT/'tianshu_prefill_int8_service_20261021/selected_plan.json').read_text());unfused_configs={r['M']:r['configuration'] for r in projection['configurations']}
  oldnative=json.loads((ROOT/'tianshu_native_int8_prefill_v2_20261022/GPU_GATE_32g.json').read_text());anchors={c['M']:c['variants']['NN_R1']['compiled']['binary_sha256'] for c in oldnative['cases'] if c['M'] in (512,4096)}
  state.update(target=str(triton.runtime.driver.active.get_current_target()),activation_original_source_sha256=sha(ROOT/'tianshu_m_range_mlp42_v2_20261021/inductor_activation.py'),activation_copied_source_sha256=sha(BASE/'inductor_activation.py'))
  assert state['activation_original_source_sha256']==state['activation_copied_source_sha256']
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint:original=torch.cat([checkpoint.get_tensor(f'model.layers.0.mlp.{r}_proj.weight') for r in ('gate','up')],0).contiguous()
  wf=original.float();scales=wf.abs().amax(1).clamp_min(1.e-8)/127.;codes=(wf/scales[:,None]).round().clamp(-127,127).to(torch.int8)
  def thash(t):return ahash(t.contiguous().view(torch.uint8).numpy())
  state['weight_signature']={key:thash(t) for key,t in [('original_BF16_sha256',original),('packed_sha256',codes),('scales_sha256',scales)]};expected=next(row for row in json.loads((ROOT/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text()) if row['prefix']=='model.layers.0.mlp.gate_up_proj');assert all(expected[k]==v for k,v in state['weight_signature'].items())
  wn,ws=codes.T.contiguous().cuda(),scales.cuda();dummy=torch.empty(1,dtype=torch.int32,device='cuda')
  def metadata(kernel):return dict(hash=kernel.hash,binary_sha256=hashlib.sha256(kernel.kernel).hexdigest(),n_regs=kernel.n_regs,n_spills=kernel.n_spills,shared=kernel.metadata.shared)
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def run_case(m):
   n,k,i=12288,2048,6144;route='native_r1' if m in anchors else 'fused' if str(m) in configurations else 'unfused';config=configurations.get(str(m)) if route=='fused' else unfused_configs[next(x for x in sorted(unfused_configs) if x>=m)] if route=='unfused' else None
   sample=np.random.default_rng(202610221200+m).standard_normal((m,k),dtype=np.float32);bits=sample.view(np.uint32);base_words=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);base_words[0]=0;ac=torch.from_numpy(base_words).view(torch.bfloat16)
   buffers=[];tensors={}
   for name,shape,dtype,value in [('a',(m,k),torch.bfloat16,17.),('q',(m,k),torch.int8,17),('asc',(m,),torch.float32,17.),('control',(m,i),torch.bfloat16,17.),('candidate',(m,i),torch.bfloat16,17.),('reference_integer',(m,n),torch.int32,-777),('reference_projected',(m,n),torch.bfloat16,17.),('integer',(m,n),torch.int32,-777),('projected',(m,n),torch.bfloat16,17.)]:
    count=int(np.prod(shape));storage=torch.full((count+512,),value,dtype=dtype,device='cuda');tensor=storage[256:count+256].view(shape);tensors[name]=tensor;buffers.append((storage,value))
   a,q,asc,integer=tensors['a'],tensors['q'],tensors['asc'],tensors['integer'];a.copy_(ac);wn.copy_(codes.T.contiguous());ws.copy_(scales)
   def epi(out,projected,diagnostics=False):return native._epilogue[(m,6)](integer,asc,ws,out,projected,m,i,1,diagnostics,num_warps=8,num_stages=1)
   def candidate():
    quant.quantize(a,q,asc);torch._int_mm(q,wn,out=integer);return epi(tensors['candidate'],tensors['candidate'])
   def control():
    quant.quantize(a,q,asc)
    if route=='native_r1':
     torch._int_mm(q,wn,out=integer);return native.epilogue(integer,asc,ws,tensors['control'],tensors['control'],1)
    if route=='fused':return fused.fused(q,wn,asc,ws,tensors['control'],dummy,tensors['control'],config)
    compiled=quant.mm(q,wn,asc,ws,tensors['reference_projected'],dummy,config);activation.triton_poi_fused_mul_silu_slice_1.run(tensors['reference_projected'],tensors['control'],m*i,stream=torch._C._cuda_getCurrentRawStream(0));return compiled
   control_kernel=control();new_kernel=candidate();torch.cuda.synchronize();control_metadata,new_metadata=metadata(control_kernel),metadata(new_kernel)
   if route=='fused':assert control_metadata['binary_sha256']==component['actual_timed_candidate_kernels'][str(m)]['binary_sha256']
   if route=='native_r1':assert control_metadata['binary_sha256']==anchors[m]==new_metadata['binary_sha256']
   graphs={}
   for name,fn in [('control',control),('candidate',candidate)]:
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn()
    graphs[name]=graph
   case=dict(M=m,K=k,N=n,current_route=route,configuration=config,current_compiled=control_metadata,candidate_compiled=new_metadata,native_workspace_bytes=m*n*4,phases=[],complete=False);state['cases'].append(case);state['phase']=f'M{m}';save()
   rows=np.unique(np.linspace(0,m-1,8).astype(np.int64));cols=np.unique(np.linspace(0,i-1,32).astype(np.int64));both=np.concatenate([cols,cols+i]);r_gpu=torch.tensor(rows,device='cuda');c_gpu=torch.tensor(cols,device='cuda');both_gpu=torch.tensor(both,device='cuda')
   for phase in range(3):
    sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.;a.copy_(ac if phase==0 else ac*(.5 if phase==1 else -.5));wn.copy_((codes*sign).T.contiguous());ws.copy_(scales*factor)
    immutable=[x.clone() for x in (a,wn,ws)];rngcpu=torch.get_rng_state().clone();rnggpu=torch.cuda.get_rng_state().clone()
    control();candidate();torch.cuda.synchronize();eager={name:words(tensors[name]) for name in graphs};eager_integer=integer.clone()
    for name,graph in graphs.items():
     tensors[name].fill_(float('nan'));integer.fill_(-2147483647);graph.replay();torch.cuda.synchronize();assert np.array_equal(words(tensors[name]),eager[name]);assert bool(torch.isfinite(tensors[name]).all())
     if name=='candidate' or route=='native_r1':assert torch.equal(integer,eager_integer)
    quant.quantize(a,q,asc)
    if route=='fused':
     fused.fused(q,wn,asc,ws,tensors['control'],tensors['reference_integer'],tensors['reference_projected'],config,diagnostics=True);assert np.array_equal(words(tensors['control']),eager['control'])
    elif route=='unfused':
     quant.mm(q,wn,asc,ws,tensors['reference_projected'],tensors['reference_integer'],config,return_int=True);activation.triton_poi_fused_mul_silu_slice_1.run(tensors['reference_projected'],tensors['control'],m*i,stream=torch._C._cuda_getCurrentRawStream(0));assert np.array_equal(words(tensors['control']),eager['control'])
    else:tensors['reference_integer'].copy_(eager_integer)
    torch._int_mm(q,wn,out=integer);assert torch.equal(integer,tensors['reference_integer']);epi(tensors['candidate'],tensors['projected'],True);torch.cuda.synchronize();assert np.array_equal(words(tensors['candidate']),eager['candidate'])
    if route=='native_r1':tensors['reference_projected'].copy_(tensors['projected'])
    assert torch.equal(tensors['projected'].view(torch.int16),tensors['reference_projected'].view(torch.int16))
    reference=tensors['control'].float();difference=(tensors['candidate'].float()-reference).abs();assert not bool((difference>(.001+.02*reference.abs())).any());mismatches=int((tensors['control'].view(torch.int16)!=tensors['candidate'].view(torch.int16)).sum())
    if route=='native_r1':assert mismatches==0
    for x,before in zip((a,wn,ws),immutable):assert torch.equal(x,before)
    assert torch.equal(rngcpu,torch.get_rng_state()) and torch.equal(rnggpu,torch.cuda.get_rng_state())
    for storage,value in buffers:assert bool((storage[:256]==value).all() and (storage[-256:]==value).all())
    artifact=root/f'M{m}_phase{phase}.npz';np.savez(artifact,rows=rows,columns=cols,integer_selected=integer[r_gpu[:,None],both_gpu[None,:]].cpu().numpy(),projected_selected_words=words(tensors['projected'][r_gpu[:,None],both_gpu[None,:]]),control_selected_words=eager['control'][np.ix_(rows,cols)],candidate_selected_words=eager['candidate'][np.ix_(rows,cols)])
    case['phases'].append(dict(phase=phase,weight_sign=sign,weight_scale_factor=factor,artifact=artifact.name,sha256=sha(artifact),A_words_sha256=ahash(words(a)),Aq_sha256=thash(q.cpu()),As_sha256=thash(asc.cpu()),WS_sha256=thash(ws.cpu()),integer_sha256=thash(integer.cpu()),projection_sha256=ahash(words(tensors['projected'])),control_output_sha256=ahash(eager['control']),candidate_output_sha256=ahash(eager['candidate']),full_integer_exact=True,full_projected_BF16_exact=True,full_output_tolerance_passed=True,full_output_BF16_bit_mismatches=mismatches,maximum_absolute_output_error=float(difference.max()),graph_poison_and_eager_equality=True,guards_inputs_weights_RNG=True));save()
   for mode in ('graph','eager'):
    for roundid in range(7):
     names=['control','candidate'] if roundid%2==0 else ['candidate','control'];times={}
     for name in names:
      fn=graphs[name].replay if mode=='graph' else control if name=='control' else candidate;begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
      for _ in range(10):fn()
      end.record();end.synchronize();times[name]=begin.elapsed_time(end)/10
     state['rounds'].append(dict(M=m,mode=mode,round=roundid,order=names,replays=10,milliseconds=times))
   for storage,value in buffers:assert bool((storage[:256]==value).all() and (storage[-256:]==value).all())
   case['complete']=True;save();print(json.dumps(dict(M=m,route=route,complete=True,bit_mismatches=[p['full_output_BF16_bit_mismatches'] for p in case['phases']])),flush=True)
  for m in range(128,4097,128):run_case(m);torch.cuda.empty_cache()
  state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_before']==state['checkpoint_sha256_after'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
