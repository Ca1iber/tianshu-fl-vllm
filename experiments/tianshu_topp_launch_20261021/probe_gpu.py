# SPDX-License-Identifier: Apache-2.0
"""Original Qrita top-p kernel: full changed-input checks and restored-input timing."""
import argparse,gc,hashlib,json,signal,statistics,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from launch import LABELS,WARPS,CASES,direct
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);a=p.parse_args();root=a.output;root.mkdir(exist_ok=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
gate=json.loads((BASE/'CPU_GATE.json').read_text());assert gate['complete']
frozen=dict(gate['sources']);frozen[str(BASE/'CPU_GATE.json')]=sha(BASE/'CPU_GATE.json')
for path,h in frozen.items():
 assert sha(path)==h
 target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==h
remaining=int((datetime.fromisoformat(a.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>0;signal.alarm(min(1800,remaining))
assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
import numpy as np
import torch,vllm_fl
import vllm.v1.sample.ops.topk_topp_triton as native
from vllm.utils.platform_utils import num_compute_units
torch.set_num_threads(8)
original_kernel=native._topk_topp_kernel
assert sha(original_kernel.fn.__code__.co_filename)==gate['sources'][original_kernel.fn.__code__.co_filename]
class Tap:
 def __init__(self,original):self.original=original;self.last=None;self.kw=None;self.grid=None
 def __getitem__(self,grid):
  def call(*args,**kwargs):self.grid=grid;self.kw=kwargs;self.last=self.original[grid](*args,**kwargs);return self.last
  return call
native._topk_topp_kernel=tap=Tap(original_kernel)
r=dict(complete=False,phase='running',model_executed=False,official_benchmark=False,adopted=False,frozen_sources=frozen,cases=[],timing_rounds=[],numerical_JIT_source_unchanged=True,only_warp_count_varies=True,timing_scope='Original top-p filtering plus full original-input GPU restore each replay; no RNG, softmax or model. Same restore in every arm; no progressively filtered logits.')
def save():
 tmp=root/'result.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(root/'result.json')
def tensor_sha(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
def metadata():
 c=tap.last;m=c.metadata
 return dict(cache_hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=getattr(c,'n_regs',None),n_spills=getattr(c,'n_spills',None),shared=getattr(m,'shared',None),num_warps=getattr(m,'num_warps',None),num_stages=getattr(m,'num_stages',None),grid=list(tap.grid),BLOCK_SIZE=tap.kw['BLOCK_SIZE'],BLOCK_SIZE_TRUNC=tap.kw['BLOCK_SIZE_TRUNC'])
def measure(graph):
 start,stop=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
 for _ in range(20):graph.replay()
 stop.record();stop.synchronize();return start.elapsed_time(stop)/20
save();generator=torch.Generator(device='cpu').manual_seed(202610211009)
try:
 for batch,kind in CASES:
  vocab=130560;programs=min(num_compute_units(0),batch);name=f'b{batch}_{kind}'
  source=torch.empty((batch,vocab),dtype=torch.float32,device='cuda')
  p_gpu=torch.empty(batch,dtype=torch.float32,device='cuda')
  buffer_storage=torch.full((programs*vocab+64,),17.,dtype=torch.float32,device='cuda');buffer=buffer_storage[32:-32].reshape(programs,vocab)
  native._TRITON_BUFFER_CACHE[(source.device,source.dtype,vocab)]=buffer
  phases=[]
  for phase in range(3):
   x=torch.randn((batch,vocab),generator=generator,dtype=torch.float32)
   x*=4. if kind in ('normal4','masked') else 1.
   if kind=='peaked':
    for row in range(batch):x[row,(row*1009+phase*37)%vocab]=32.+row%4
   if kind=='uniform':x.zero_()
   if kind=='masked':x[:,(torch.arange(vocab)+phase)%19==0]=-float('inf')
   if kind=='nonfinite':
    x[0].fill_(-float('inf'));x[1,17]=float('nan');x[2,33]=float('inf');x[3,4:]=float('nan')
   x=x.to(torch.bfloat16).contiguous();xc=x.float();ps=torch.full((batch,),.95,dtype=torch.float32)
   if phase==2:ps[:6]=torch.tensor([0.,.1,.5,.9,.9999,1.])
   phases.append(dict(phase=phase,input_bf16=x,p=ps))
  source.copy_(phases[0]['input_bf16']);p_gpu.copy_(phases[0]['p'])
  native.apply_top_k_top_p_triton(source.clone(),None,p_gpu);tables=native._TRITON_TABLE_CACHE[source.device]
  works={};graphs={};compiled={}
  for label in LABELS:
   storage=torch.full((batch,vocab+7),17.,dtype=torch.float32,device='cuda');work=storage[:,:vocab];works[label]=(work,storage)
   def invoke():
    work.copy_(source)
    if label=='native':native.apply_top_k_top_p_triton(work,None,p_gpu)
    else:direct(native,work,p_gpu,buffer,tables,programs,WARPS[label])
   for _ in range(2):invoke()
   torch.cuda.synchronize();compiled[label]=metadata()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):invoke()
   graphs[label]=graph
  assert compiled['native']['BLOCK_SIZE']==8192 and compiled['native']['BLOCK_SIZE_TRUNC']==4096
  assert compiled['native']['num_warps']==4,compiled['native']
  assert compiled['warp4']['binary_sha256']==compiled['native']['binary_sha256']
  case=dict(name=name,batch=batch,vocab=vocab,kind=kind,programs=programs,compiled=compiled,phases=[]);r['cases'].append(case);save()
  for data in phases:
   phase=data['phase'];xc=data['input_bf16'].float();ps=data['p'];source.copy_(xc);p_gpu.copy_(ps)
   cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();outputs={};native_out=None
   for label in LABELS:
    graphs[label].replay();torch.cuda.synchronize();work,storage=works[label];actual=work.cpu()
    keep=~torch.isneginf(actual);packed=np.packbits(keep.numpy().reshape(-1),bitorder='little').copy()
    expected=xc.clone();expected[~keep]=-float('inf')
    assert torch.equal(actual.view(torch.int32),expected.view(torch.int32)),(name,phase,label,'retained_values_changed')
    if label=='native':native_out=actual
    equal=torch.equal(actual.view(torch.int32),native_out.view(torch.int32));different=int((actual.view(torch.int32)!=native_out.view(torch.int32)).sum())
    assert bool((storage[:,vocab:]==17).all()) and bool((buffer_storage[:32]==17).all()) and bool((buffer_storage[-32:]==17).all())
    assert torch.equal(source.cpu().view(torch.int32),xc.view(torch.int32)) and torch.equal(p_gpu.cpu(),ps)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    outputs[label]=dict(keep_mask_packbits_little=torch.from_numpy(packed),full_output_sha256=tensor_sha(actual),full_output_bitwise_native=equal,different_elements=different,retained_values_bitwise_original=True,guards_preserved=True,inputs_preserved=True,RNG_unchanged=True)
   archive=dict(input_bf16=data['input_bf16'],p=ps,outputs=outputs,name=name,phase=phase)
   path=root/f'{name}_phase{phase}.pt';torch.save(archive,path)
   case['phases'].append(dict(phase=phase,archive=path.name,sha256=sha(path),comparisons={k:{z:v for z,v in out.items() if z!='keep_mask_packbits_little'} for k,out in outputs.items()}));save()
  # Replays restore this fixed *unfiltered* source inside every graph.
  source.copy_(phases[0]['input_bf16']);p_gpu.copy_(phases[0]['p']);before=source.cpu().clone();p_before=p_gpu.cpu().clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
  for round_index in range(7):
   order=list(LABELS[round_index%len(LABELS):]+LABELS[:round_index%len(LABELS)])
   if round_index%2:order.reverse()
   for label in order:r['timing_rounds'].append(dict(case=name,round=round_index,label=label,milliseconds=measure(graphs[label])))
  assert torch.equal(source.cpu().view(torch.int32),before.view(torch.int32)) and torch.equal(p_gpu.cpu(),p_before)
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
  assert bool((buffer_storage[:32]==17).all()) and bool((buffer_storage[-32:]==17).all()) and all(bool((s[:,vocab:]==17).all()) for w,s in works.values())
  case['timed_input_RNG_and_guards_preserved']=True
  case['median_ms']={label:statistics.median(x['milliseconds'] for x in r['timing_rounds'] if x['case']==name and x['label']==label) for label in LABELS}
  print(json.dumps(dict(case=name,median_ms=case['median_ms'],all_phases_bitwise_native={label:all(x['comparisons'][label]['full_output_bitwise_native'] for x in case['phases']) for label in LABELS})),flush=True);save()
  graphs.clear();works.clear();native._TRITON_BUFFER_CACHE.clear();del buffer,buffer_storage,source,p_gpu;gc.collect();torch.cuda.empty_cache()
 r.update(complete=True,phase='complete')
except BaseException as error:r.update(phase='failed',error=repr(error));save();raise
save();print(json.dumps(dict(complete=True,cases=len(r['cases']),timing_trials=len(r['timing_rounds']))),flush=True)
