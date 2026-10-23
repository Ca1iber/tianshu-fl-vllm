# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json,os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261023_IR_RMS_A8_chain_GPU_tianshu')
assert not ROOT.exists();ROOT.mkdir()
os.environ['TRITON_CACHE_DIR']=str(ROOT/'triton_cache');os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'inductor_cache')
import numpy as np
import torch
from vllm.ir.ops import layernorm as ir

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
prep=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());assert prep['complete']
for p,h in prep['sources'].items():
 assert sha(p)==h;t=ROOT/'source_snapshot'/Path(p).relative_to('/');t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(Path(p).read_bytes());assert sha(t)==h
prior=json.loads(Path(prep['prior_result']).read_text());priorgate=json.loads(Path(prep['prior_gate']).read_text());assert priorgate['complete'] and not priorgate['component_lead'] and len(prior['cases'])==108
kernel=load('_IR_RMS_A8_chain_same_kernel',EXP/'tianshu_IR_RMS_A8_20261023/kernel.py');dense=load('_IR_RMS_A8_chain_old_quantizer',EXP/'tianshu_dense_w8a8_20261020/kernel.py');torch.set_num_threads(4)
native_norm=ir.rms_norm.impls['native'].impl_fn;native_add=ir.fused_add_rms_norm.impls['native'].impl_fn

def original_norm(x,residual,weight):
 if residual is None:return native_norm(x,weight,1.e-6,None),None
 return native_add(x,residual,weight,1.e-6,None)
compiled_norm=torch.compile(original_norm,backend='inductor',fullgraph=True,dynamic=False)

def tbytes(t):return t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
def tensor(name):
 e=prior['files'][name];assert sha(e['path'])==e['sha256'];data=np.fromfile(e['path'],dtype=np.uint16).reshape(e['shape']);return torch.from_numpy(data.copy()).view(torch.bfloat16).cuda()
compiled={};records=[];timings=[];files={}
def call(kind,x,r,w):
 m,k=x.shape;q=torch.empty_like(x,dtype=torch.int8);s=torch.empty((m,),device=x.device,dtype=torch.float32)
 if kind=='compiled_IR_then_A8':y,rr=compiled_norm(x,r,w);dense.quantize(y,q,s)
 else:
  y=torch.empty_like(x);rr=torch.empty_like(x) if r is not None else None;c=kernel.run(x,r,w,y,rr,q,s);key=str(r is not None)
  if key not in compiled:
   item=dict(binary_sha256=hashlib.sha256(c.kernel).hexdigest(),cache_hash=c.hash,n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared);assert item==prior['candidate_compiled'][key];compiled[key]=item
 return dict(norm=y,residual=rr,q=q,scale=s)

def save(name,t):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tbytes(t));files[name]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size);return name

for M in (8,64,128):
 for has in (False,True):
  case=next(c for c in prior['cases'] if c['bank']==0 and c['M']==M and c['has_residual']==has and c['phase']==0);x=tensor(case['input_files']['x']);r=tensor(case['input_files']['residual']) if has else None;w=tensor(case['input_files']['weight']);before=[tbytes(t) for t in (x,r,w) if t is not None];graphs={}
  for kind in ('compiled_IR_then_A8','fused_IR_A8'):
   for _ in range(2):call(kind,x,r,w)
   torch.cuda.synchronize()
   for chain in (1,32,128):
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
     for _ in range(chain):outputs=call(kind,x,r,w)
    for t in outputs.values():
     if t is not None:t.fill_(99 if t.dtype==torch.int8 else float('nan'))
    g.replay();torch.cuda.synchronize();oldgroup=case['control' if kind=='compiled_IR_then_A8' else 'candidate'];saved={}
    for field,t in outputs.items():
     if t is not None:
      assert tbytes(t)==Path(prior['files'][oldgroup[field]]['path']).read_bytes(),(M,has,kind,chain,field)
      saved[field]=save(f'graphs/M{M}_R{int(has)}_{kind}_L{chain}_{field}.bin',t)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state()) and before==[tbytes(t) for t in (x,r,w) if t is not None]
    graphs[kind,chain]=(g,outputs);records.append(dict(M=M,has_residual=has,kind=kind,chain_length=chain,actual_graph_captured_calls=chain,prior_case=case['case'],prior_group='control' if kind=='compiled_IR_then_A8' else 'candidate',saved_outputs=saved,poison_replay_exact_to_prior_complete_outputs=True,input_weight_RNG_unchanged=True))
  for chain in (1,32,128):
   for repeat in range(5):
    kinds=('compiled_IR_then_A8','fused_IR_A8') if repeat%2==0 else ('fused_IR_A8','compiled_IR_then_A8')
    for kind in kinds:
     g,_=graphs[kind,chain]
     for _ in range(10):g.replay()
     torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
     for _ in range(20):g.replay()
     end.record();end.synchronize();elapsed=start.elapsed_time(end);timings.append(dict(M=M,has_residual=has,chain_length=chain,kind=kind,round=repeat+1,excluded=repeat==0,order_within_round=kinds.index(kind),warm_replays=10,timed_replays=20,total_graph_ms=elapsed,ms_per_component=elapsed/(20*chain)))
  del graphs
assert len(records)==36 and len(timings)==180
result=dict(complete=True,component_only=True,measurement_method_changed=True,numerical_kernel_unchanged=True,inherited_108_numeric_cases_from=prep['prior_gate'],inherited_numeric_gate_sha256=prep['sources'][prep['prior_gate']],prior_result=prep['prior_result'],prior_result_sha256=prep['sources'][prep['prior_result']],sources=prep['sources'],graphs=records,timings=timings,files=files,candidate_compiled=compiled,original_checkpoint_sha256=priorgate['checkpoint_sha256'],no_full_model_CUBIN_equivalence_claim=True,no_model_quality_or_perf_claim=True,adopted=False)
(ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('sources','graphs','timings','files')}))
