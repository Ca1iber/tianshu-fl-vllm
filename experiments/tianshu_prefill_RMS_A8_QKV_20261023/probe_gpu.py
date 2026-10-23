# SPDX-License-Identifier: Apache-2.0
"""Frozen complete-chain component probe; run only after all model jobs finish."""
import argparse,hashlib,importlib.util,json,os,signal,struct,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
KINDS=('compiled_IR_A8_native_QKV','fused_IR_A8_native_QKV')
FIELDS=('norm','residual','q','scale','integer','projection')

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def encode(values):
 bits=np.asarray(values,dtype=np.float32).view(np.uint32);return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
def inputs(M,layer,phase,has):
 rng=np.random.default_rng(202610231400+100*layer+10*phase+int(has));x=rng.standard_normal((M,2048),dtype=np.float32);r=rng.standard_normal((M,2048),dtype=np.float32);factor=np.float32((1.,.0625,1e-5)[phase]);x=encode(x*factor);r=encode(r*factor);x[0]=0;r[0]=0
 if phase==1 and has:r[1]=x[1]^np.uint16(0x8000)
 return x,r if has else None

def sample_indices(index,M):
 rng=np.random.default_rng(202610231700+index);rows=[0,1,*sorted(rng.choice(np.arange(2,M-1),5,replace=False).tolist()),M-1];cols=[0,1,*sorted(rng.choice(np.arange(2,2559),13,replace=False).tolist()),2559];assert len(set(rows))==8 and len(set(cols))==16;return rows,cols

def tensor_bytes(tensor):return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
def tensor_sha(tensor):return hashlib.sha256(tensor_bytes(tensor)).hexdigest()
def guarded_from_bits(bits):
 shape=tuple(bits.shape);count=int(bits.size);raw=torch.full((count+256,),31.,device='cuda',dtype=torch.bfloat16);tensor=raw[128:128+count].view(shape);tensor.copy_(torch.from_numpy(np.array(bits,copy=True)).view(torch.bfloat16).cuda());return tensor,(raw,count,31.)
def assert_equal(left,right):
 assert set(left)==set(right)==set(FIELDS)
 for field in FIELDS:
  if left[field] is None:assert right[field] is None
  else:assert right[field] is not None and torch.equal(left[field].view(torch.uint8),right[field].view(torch.uint8)),field

def poison(outputs):
 for tensor in outputs.values():
  if tensor is not None:tensor.fill_(99 if tensor.dtype==torch.int8 else 1245391901 if tensor.dtype==torch.int32 else float('nan'))

def capture_checked(runtime,kind,x,residual,bank,eager,chain=1,guarded=True):
 for _ in range(2):runtime(kind,x,residual,bank['norm'],bank['packed'],bank['scale'],guarded)
 torch.cuda.synchronize();graph=torch.cuda.CUDAGraph();count_before=sum(runtime.calls.values())
 with torch.cuda.graph(graph):
  for _ in range(chain):captured,guards=runtime(kind,x,residual,bank['norm'],bank['packed'],bank['scale'],guarded)
 assert sum(runtime.calls.values())-count_before==chain
 poison(captured);graph.replay();torch.cuda.synchronize();assert_equal(eager,captured);runtime.check_guards(guards)
 return graph,captured,guards

class RawFiles:
 def __init__(self,root):self.root=root;self.files={}
 def save(self,name,tensor,link_to=None):
  payload=tensor_bytes(tensor);path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256(payload).hexdigest();assert not path.exists()
  if link_to is None:path.write_bytes(payload)
  else:
   entry=self.files[link_to];assert entry['bytes']==len(payload) and entry['sha256']==digest;assert payload==(self.root/link_to).read_bytes();os.link(self.root/link_to,path)
  self.files[name]=dict(path=str(path),sha256=digest,bytes=len(payload),shape=list(tensor.shape),dtype=str(tensor.dtype),hardlink_to=link_to);return name

def main():
 global torch
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
 remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200
 state=dict(complete=False,phase='preparing',component_only=True,official_benchmark=False,adopted=False,cases=[],timings=[],timing_graphs=[],banks=[],files={},stop_at_utc=args.stop_at_utc)
 raw=None
 def save_state():
  state['files']=raw.files if raw is not None else state['files'];(root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 def stop(signum,frame):raise InterruptedError(f'signal {signum}')
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,stop)
 signal.alarm(remaining)
 try:
  sources={}
  for line in (BASE/'GPU_INPUTS.sha256').read_text().splitlines():
   digest,path=line.split('  ',1);assert sha(path)==digest;sources[path]=digest
  sources[str(BASE/'GPU_INPUTS.sha256')]=sha(BASE/'GPU_INPUTS.sha256');state['sources']=sources
  for path,digest in sources.items():
   target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==digest
  preparation=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());assert preparation['complete'] and preparation['expected_numeric_cases']==18 and preparation['three_QKV_weight_packs_independently_recomputed_exact']
  for path,digest in preparation['sources'].items():assert sources[path]==digest==sha(path)
  hardware=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_before.txt').write_text(hardware);assert 'No running processes found' in hardware
  os.environ['TRITON_CACHE_DIR']=str(root/'triton_cache');os.environ['TORCHINDUCTOR_CACHE_DIR']=str(root/'inductor_cache')
  import torch
  torch.set_num_threads(4);chain=load('_prefill_RMS_QKV_complete_chain',BASE/'chain.py');runtime=chain.Runtime();packer=load('_prefill_RMS_QKV_original_packer',EXP/'tianshu_w8_gpu_packing_20261020/packing.py');raw=RawFiles(root);state.update(phase='numeric',torch_version=torch.__version__,numpy_version=np.__version__,model_checkpoint_sha256=preparation['checkpoint_sha256']);save_state();banks=[];fixtures=[]
  with MODEL.open('rb') as checkpoint:
   length=struct.unpack('<Q',checkpoint.read(8))[0];header=json.loads(checkpoint.read(length));offset=8+length
   def weight_bits(name,shape):
    meta=header[name];assert meta['dtype']=='BF16' and meta['shape']==list(shape);lo,hi=meta['data_offsets'];assert hi-lo==int(np.prod(shape))*2;checkpoint.seek(offset+lo);payload=checkpoint.read(hi-lo);assert len(payload)==hi-lo;return np.frombuffer(payload,dtype='<u2').reshape(shape)
   for proof in preparation['selected_banks']:
    layer=proof['layer'];bits=np.concatenate([weight_bits(f'model.layers.{layer}.self_attn.{role}_proj.weight',(n,2048)) for role,n in (('q',2048),('k',256),('v',256))],axis=0);weight,weight_guard=guarded_from_bits(bits);assert tensor_sha(weight)==proof['original_BF16_sha256'];norm_bits=np.fromfile(proof['norm_path'],dtype='<u2');assert hashlib.sha256(norm_bits.tobytes()).hexdigest()==proof['norm_sha256'];norm,norm_guard=guarded_from_bits(norm_bits)
    nt=torch.empty((2560,2048),device='cuda',dtype=torch.int8);scale=torch.empty((2560,),device='cuda',dtype=torch.float32);packer.pack_into(weight,nt,scale,runtime.dense);packed=nt.T.contiguous();torch.cuda.synchronize();assert tensor_sha(nt)==proof['packed_NT_sha256'] and tensor_sha(scale)==proof['scale_sha256'];bank=dict(proof=proof,weight=weight,norm=norm,packed=packed,scale=scale,guards=[weight_guard,norm_guard]);bank['versions']=[int(t._version) for t in (weight,norm,packed,scale)];bank['hashes']=[tensor_sha(t) for t in (weight,norm,packed,scale)];banks.append(bank)
    state['banks'].append(dict(layer=layer,has_residual=proof['has_residual'],original_BF16_sha256=tensor_sha(weight),packed_NT=raw.save(f'banks/{layer}/packed_nt.i8',nt),scale=raw.save(f'banks/{layer}/scale.f32',scale),norm=raw.save(f'banks/{layer}/norm.u16',norm),original_packs_exact=True));del nt,bits
  for bank in banks:
   layer=bank['proof']['layer'];has=bank['proof']['has_residual']
   for M in preparation['M_values']:
    xb,rb=inputs(M,layer,0,has);warmx,_=guarded_from_bits(xb);warmr=guarded_from_bits(rb)[0] if has else None
    for kind in KINDS:
     for _ in range(2):runtime(kind,warmx,warmr,bank['norm'],bank['packed'],bank['scale'])
    torch.cuda.synchronize();del warmx,warmr
    for phase in preparation['phases']:
     xb,rb=inputs(M,layer,phase,has);x,xguard=guarded_from_bits(xb);r,rguard=guarded_from_bits(rb) if has else (None,None);before=[tensor_sha(t) for t in (x,r) if t is not None];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();index=len(state['cases']);prefix=f'cases/{index:03d}';rows,cols=sample_indices(index,M);row_index=torch.tensor(rows,device='cuda');col_index=torch.tensor(cols,device='cuda');case=dict(case=index,layer=layer,M=M,has_residual=has,phase=phase,input_seed=202610231400+100*layer+10*phase+int(has),projection_sample_rows=rows,projection_sample_columns=cols,input_files={'x':raw.save(prefix+'/x.u16',x)},control={},candidate={},control_graph={},candidate_graph={},full_integer_sha256={},integer_samples={})
     if has:case['input_files']['residual']=raw.save(prefix+'/residual.u16',r)
     for kind,group in zip(KINDS,('control','candidate')):
      eager,guards=runtime(kind,x,r,bank['norm'],bank['packed'],bank['scale'],True);torch.cuda.synchronize();runtime.check_guards(guards);graph,captured,graph_guards=capture_checked(runtime,kind,x,r,bank,eager)
      for field in FIELDS:
       if eager[field] is None:continue
       if field=='integer':
        eh=tensor_sha(eager[field]);gh=tensor_sha(captured[field]);assert eh==gh;case['full_integer_sha256'][group]=dict(eager=eh,graph=gh,shape=[M,2560],full_GPU_bytes_compared=True);sample=eager[field][row_index[:,None],col_index[None,:]];graph_sample=captured[field][row_index[:,None],col_index[None,:]];name=raw.save(prefix+f'/{group}_integer_sample.i32',sample);gname=raw.save(prefix+f'/{group}_graph_integer_sample.i32',graph_sample,name);case['integer_samples'][group]=dict(eager=name,graph=gname)
       else:
        name=raw.save(prefix+f'/{group}_{field}.bin',eager[field]);case[group][field]=name;case[group+'_graph'][field]=raw.save(prefix+f'/{group}_graph_{field}.bin',captured[field],name)
      runtime.check_guards(guards+graph_guards);del graph,eager,captured
     runtime.check_guards([xguard]+([rguard] if has else [])+bank['guards']);assert before==[tensor_sha(t) for t in (x,r) if t is not None];assert bank['versions']==[int(t._version) for t in (bank['weight'],bank['norm'],bank['packed'],bank['scale'])] and bank['hashes']==[tensor_sha(t) for t in (bank['weight'],bank['norm'],bank['packed'],bank['scale'])];assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());case.update(actual_Graph_return_objects_full_bytes_poison_replay_exact=True,input_weights_RNG_and_all_guardable_outputs_passed=True);state['cases'].append(case);state['files']=raw.files;save_state()
     if phase==0:fixtures.append(dict(case=index,bank=bank,M=M,x=x,residual=r))
  assert len(state['cases'])==18 and len(fixtures)==6
  state['phase']='timing';save_state()
  for fixture in fixtures:
   case=state['cases'][fixture['case']];bank=fixture['bank'];x=fixture['x'];r=fixture['residual'];M=fixture['M'];graphs={};before_inputs=[tensor_sha(t) for t in (x,r) if t is not None];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
   for kind,group in zip(KINDS,('control','candidate')):
    reference,_=runtime(kind,x,r,bank['norm'],bank['packed'],bank['scale']);torch.cuda.synchronize();expected={field:raw.files[name]['sha256'] for field,name in case[group].items()};expected['integer']=case['full_integer_sha256'][group]['eager'];assert {field:tensor_sha(tensor) for field,tensor in reference.items() if tensor is not None}==expected
    for chain_length in (4,16):
     graph,captured,guards=capture_checked(runtime,kind,x,r,bank,reference,chain_length,False);actual={field:tensor_sha(tensor) for field,tensor in captured.items() if tensor is not None};assert actual==expected;graphs[kind,chain_length]=(graph,captured);state['timing_graphs'].append(dict(case=case['case'],layer=case['layer'],M=M,has_residual=case['has_residual'],kind=kind,chain_length=chain_length,actual_graph_captured_full_chain_calls=chain_length,actual_Graph_return_objects_poison_replay_exact=True,outputs_sha256=actual,integer_full_bytes_compared=True))
    del reference
   for execution,chain_length in (('eager',1),('graph',4),('graph',16)):
    for repeat in range(1,6):
     order=KINDS if repeat%2 else KINDS[::-1]
     for position,kind in enumerate(order):
      if execution=='eager':fn=lambda kind=kind:runtime(kind,x,r,bank['norm'],bank['packed'],bank['scale'])
      else:fn=graphs[kind,chain_length][0].replay
      for _ in range(10):fn()
      start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
      for _ in range(20):fn()
      end.record();end.synchronize();total=float(start.elapsed_time(end));state['timings'].append(dict(case=case['case'],layer=case['layer'],M=M,has_residual=case['has_residual'],execution=execution,chain_length=chain_length,kind=kind,round=repeat,excluded=repeat==1,order_within_round=position,warm_replays=10,timed_replays=20,total_event_ms=total,ms_per_complete_chain=total/(20*chain_length)))
   assert before_inputs==[tensor_sha(t) for t in (x,r) if t is not None];assert bank['versions']==[int(t._version) for t in (bank['weight'],bank['norm'],bank['packed'],bank['scale'])] and bank['hashes']==[tensor_sha(t) for t in (bank['weight'],bank['norm'],bank['packed'],bank['scale'])];assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());del graphs;save_state()
  assert len(state['timings'])==180 and len(state['timing_graphs'])==24
  for path,digest in sources.items():assert sha(path)==digest
  state.update(complete=True,phase='complete',compiled=runtime.compiled,operator_calls=runtime.calls,operator_calls_are_host_invocations_not_graph_replay_counts=True,numerical_kernel_unchanged=True,all_actual_Graph_outputs_full_bytes_checked=True,input_weight_RNG_and_guards_passed=True,standalone_compiled_IR_baseline=True,no_full_model_CUBIN_equivalence_claim=True,fresh_model_quality_required=True)
 except BaseException as error:
  state.update(complete=False,phase='failed',error=repr(error));raise
 finally:
  signal.alarm(0)
  if raw is not None:state['files']=raw.files
  (root/'state.json').write_text(json.dumps(state,indent=2)+'\n');(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
 print(json.dumps({key:value for key,value in state.items() if key not in ('cases','timings','timing_graphs','sources','files','banks','operator_calls')}))

if __name__=='__main__':main()
