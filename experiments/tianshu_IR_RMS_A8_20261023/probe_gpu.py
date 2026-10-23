# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,importlib.util,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--stop-at-utc',required=True);args=p.parse_args();ROOT=args.output.resolve();BASE=Path(__file__).resolve().parent;EXP=BASE.parent
assert ROOT.is_relative_to('/workspace/benchmark-results') and not ROOT.exists();ROOT.mkdir();assert (datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds()>600
os.environ['TRITON_CACHE_DIR']=str(ROOT/'triton_cache');os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'inductor_cache')
import numpy as np
import torch
from vllm.ir.ops import layernorm as ir

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
prepare=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());assert prepare['complete'] and prepare['expected_numeric_cases']==108
for path,h in prepare['sources'].items():assert sha(path)==h
for item in prepare['selected_norms']:assert sha(item['path'])==item['sha256']
(ROOT/'sources.json').write_text(json.dumps(prepare['sources'],indent=2)+'\n')
for path,h in prepare['sources'].items():
 target=ROOT/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==h
candidate=load('_IR_RMS_A8_fixed_kernel',BASE/'kernel.py');dense=load('_IR_RMS_A8_old_quantizer',EXP/'tianshu_dense_w8a8_20261020/kernel.py');torch.set_num_threads(4)
native_norm=ir.rms_norm.impls['native'].impl_fn;native_add=ir.fused_add_rms_norm.impls['native'].impl_fn
assert Path(native_norm.__code__.co_filename).resolve()==Path('/usr/local/lib/python3.12/site-packages/vllm/ir/ops/layernorm.py')
def original_norm(x,residual,weight):
 if residual is None:return native_norm(x,weight,1.e-6,None),None
 return native_add(x,residual,weight,1.e-6,None)
compiled_norm=torch.compile(original_norm,backend='inductor',fullgraph=True,dynamic=False)

def bf16_bits(value):
 bits=np.asarray(value,dtype=np.float32).view(np.uint32);return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
def inputs(M,has,phase,bank):
 rng=np.random.default_rng(202610230700+100*bank+10*phase+int(has));x=rng.standard_normal((M,2048),dtype=np.float32);r=rng.standard_normal((M,2048),dtype=np.float32)
 scale=(1.,.0625,1.e-5)[phase];x=bf16_bits(x*np.float32(scale));r=bf16_bits(r*np.float32(scale));x[0]=0;r[0]=0
 if phase==1 and M>1:r[1]=x[1]^np.uint16(0x8000)
 return x,r if has else None

def guarded(shape,dtype):
 count=int(np.prod(shape));tag=71 if dtype==torch.int8 else 31.;raw=torch.full((count+256,),tag,device='cuda',dtype=dtype);return raw[128:128+count].view(shape),(raw,count,tag)
def check_guards(items):
 for raw,count,tag in items:assert bool((raw[:128]==tag).all()) and bool((raw[128+count:]==tag).all())
compiled={};files={};cases=[];timings=[];rawcache={}
def call(kind,x,residual,weight,guards=True):
 m,k=x.shape;protect=[]
 def alloc(shape,dtype):
  if not guards:return torch.empty(shape,device='cuda',dtype=dtype)
  view,record=guarded(shape,dtype);protect.append(record);return view
 q=alloc((m,k),torch.int8);scale=alloc((m,),torch.float32)
 if kind=='compiled_IR_then_A8':y,r=compiled_norm(x,residual,weight);dense.quantize(y,q,scale)
 else:
  y=alloc((m,k),torch.bfloat16);r=alloc((m,k),torch.bfloat16) if residual is not None else None;kernel=candidate.run(x,residual,weight,y,r,q,scale);key=str(residual is not None)
  if guards or key not in compiled:
   meta=dict(binary_sha256=hashlib.sha256(kernel.kernel).hexdigest(),cache_hash=kernel.hash,n_regs=kernel.n_regs,n_spills=kernel.n_spills,shared=kernel.metadata.shared)
   if key in compiled:assert compiled[key]==meta
   else:compiled[key]=meta
 return dict(norm=y,residual=r,q=q,scale=scale),protect

def tensor_bytes(t):return t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
def save(name,tensor):
 path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(tensor_bytes(tensor));entry=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,shape=list(tensor.shape),dtype=str(tensor.dtype));files[name]=entry;return name

def equal(a,b):
 assert set(a)==set(b)
 for key in a:
  if a[key] is None:assert b[key] is None
  else:assert torch.equal(a[key],b[key]),key

def graph_checked(kind,x,residual,weight,eager):
 for _ in range(2):call(kind,x,residual,weight)
 torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):captured,guards=call(kind,x,residual,weight)
 for tensor in captured.values():
  if tensor is not None:tensor.fill_(99 if tensor.dtype==torch.int8 else float('nan'))
 graph.replay();torch.cuda.synchronize();equal(eager,captured);check_guards(guards)
 return captured

for bank,item in enumerate(prepare['selected_norms']):
 wbits=np.fromfile(item['path'],dtype=np.uint16);assert wbits.shape==(2048,);weight=torch.from_numpy(wbits.copy()).view(torch.bfloat16).cuda();before_weight=tensor_bytes(weight);version=int(weight._version)
 for M in prepare['M_values']:
  for has in (False,True):
   # Resolve and compile both before taking the per-case RNG checkpoint.
   sample_x,sample_r=inputs(M,has,0,bank);warmx=torch.from_numpy(sample_x.copy()).view(torch.bfloat16).cuda();warmr=torch.from_numpy(sample_r.copy()).view(torch.bfloat16).cuda() if has else None
   for kind in ('compiled_IR_then_A8','fused_IR_A8'):call(kind,warmx,warmr,weight)
   torch.cuda.synchronize()
   for phase in prepare['phases']:
    bits,r_bits=inputs(M,has,phase,bank);x=torch.from_numpy(bits.copy()).view(torch.bfloat16).cuda();r=torch.from_numpy(r_bits.copy()).view(torch.bfloat16).cuda() if has else None;before_x=tensor_bytes(x);before_r=tensor_bytes(r) if has else None;cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();control,cg=call('compiled_IR_then_A8',x,r,weight);fused,fg=call('fused_IR_A8',x,r,weight);torch.cuda.synchronize();check_guards(cg+fg);control_graph=graph_checked('compiled_IR_then_A8',x,r,weight,control);fused_graph=graph_checked('fused_IR_A8',x,r,weight,fused)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state()) and before_x==tensor_bytes(x) and (not has or before_r==tensor_bytes(r)) and before_weight==tensor_bytes(weight) and int(weight._version)==version
    index=len(cases);prefix=f'cases/case{index:03d}';row=dict(case=index,bank=bank,bank_name=item['name'],M=M,has_residual=has,phase=phase,input_seed=202610230700+100*bank+10*phase+int(has),input_files={'x':save(prefix+'/x.u16',x),'weight':save(prefix+'/weight.u16',weight)},control={},candidate={},control_graph={},candidate_graph={},actual_Graph_return_objects_poison_replay_exact=True,all_output_guards_passed=True,input_weight_RNG_preserved=True)
    if has:row['input_files']['residual']=save(prefix+'/residual.u16',r)
    for group,outputs in (('control',control),('candidate',fused),('control_graph',control_graph),('candidate_graph',fused_graph)):
     for name,tensor in outputs.items():
      if tensor is not None:row[group][name]=save(prefix+f'/{group}_{name}.bin',tensor)
    cases.append(row)
 # Full selected bank used; only bank0 gets diagnostic timing at all M/residual.
 if bank==0:
  for M in prepare['M_values']:
   for has in (False,True):
    xb,rb=inputs(M,has,0,bank);x=torch.from_numpy(xb.copy()).view(torch.bfloat16).cuda();r=torch.from_numpy(rb.copy()).view(torch.bfloat16).cuda() if has else None
    for kind in ('compiled_IR_then_A8','fused_IR_A8'):
     for _ in range(2):call(kind,x,r,weight,False)
     torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):retained,_=call(kind,x,r,weight,False)
     for execution in ('eager','graph'):
      fn=(lambda:call(kind,x,r,weight,False)) if execution=='eager' else graph.replay
      for repeat in range(5):
       for _ in range(20):fn()
       start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
       for _ in range(200):fn()
       end.record();end.synchronize();timings.append(dict(M=M,has_residual=has,kind=kind,execution=execution,round=repeat+1,excluded=repeat==0,ms=start.elapsed_time(end)/200))
assert len(cases)==108 and len(timings)==120
for path,h in prepare['sources'].items():assert sha(path)==h
result=dict(complete=True,component_only=True,standalone_compiled_IR_baseline=True,actual_full_model_CUBIN_equivalence_not_asserted=True,fresh_model_quality_still_required=True,cases=cases,timings=timings,files=files,candidate_compiled=compiled,model_checkpoint_sha256=prepare['checkpoint_sha256'],sources=prepare['sources'],torch_version=torch.__version__,numpy_version=np.__version__,GPU_RNG_preserved_per_case=True,original_inputs_weights_preserved=True,Graph_return_objects_poison_replay_all108_passed=True,all_guards_passed=True,adopted=False,official_benchmark=False)
(ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('cases','timings','files','sources')}))
