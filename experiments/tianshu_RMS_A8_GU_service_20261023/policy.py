# SPDX-License-Identifier: Apache-2.0
import atexit,functools,hashlib,importlib.util,json,os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def install(worker,mode,directory):
 import torch
 from vllm_fl.worker.model_runner import ModelRunnerFL
 assert mode in ('control','candidate');cfg=worker.vllm_config;assert cfg.model_config.model=='/workspace/MiniCPM5-2B' and cfg.speculative_config is None and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.cache_config.kv_cache_memory_bytes is None
 comp=EXP/'tianshu_IR_RMS_A8_20261023';numeric=json.loads((comp/'INDEPENDENT_CPU_GATE_32g.json').read_text());lead=json.loads((EXP/'tianshu_IR_RMS_A8_chain_20261023/INDEPENDENT_CPU_GATE_32g.json').read_text());assert numeric['complete'] and numeric['numeric_cases']==108 and lead['complete'] and lead['component_lead'] and lead['numerical_kernel_unchanged_and_CUBIN_exact']
 norm=load('_RMS_A8_GU_fixed_norm',comp/'kernel.py');gu=load('_RMS_A8_GU_original_GEMM',EXP/'tianshu_nn_silu_two_dot_20261021/kernel.py');nn=sys.modules['_nn_silu_fx_policy']
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);config=nn.RECORD['configuration'];assert config==dict(bm=64,bn=64,bk=128,warps=4,stages=1,mode='two_dot')
 record=dict(installed=True,mode=mode,completed=False,all42_new_norm_capture_verified=False,routes={},warm_before_capture_RNG_unchanged=False,source=str(BASE/'policy.py'),source_sha256=sha(__file__),norm_source=str(comp/'kernel.py'),norm_sha256=sha(comp/'kernel.py'),GEMM_source=str(EXP/'tianshu_nn_silu_two_dot_20261021/kernel.py'),GEMM_sha256=sha(EXP/'tianshu_nn_silu_two_dot_20261021/kernel.py'),component_numeric_gate_sha256=sha(comp/'INDEPENDENT_CPU_GATE_32g.json'),component_chain_gate_sha256=sha(EXP/'tianshu_IR_RMS_A8_chain_20261023/INDEPENDENT_CPU_GATE_32g.json'),expected_norm_binary=numeric['compiled_candidate_kernels']['True']['binary_sha256'],expected_GEMM_binary=nn.RECORD['component_expected_binary_sha256'],no_new_packed_weight_payload=True,no_global_tensor_workspace=True,QKV_Down_LM_routes_unchanged=True,old_compatibility_index_is_actual_GEMM_kernel_calls_not_old_custom_op_or_A8_calls=True,Python_calls_are_not_Graph_replays=True)
 def save():
  p=directory/f'route_{os.getpid()}.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(record,indent=2)+'\n');t.replace(p)
 @torch.library.custom_op('tianshu_RMS_A8_GU::fused',mutates_args=(),device_types='cuda')
 def fused(x:torch.Tensor,residual:torch.Tensor,norm_weight:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->tuple[torch.Tensor,torch.Tensor]:
  assert mode=='candidate' and x.shape==residual.shape==(64,2048) and x.dtype==residual.dtype==torch.bfloat16 and x.is_contiguous() and residual.is_contiguous() and norm_weight.shape==(2048,) and norm_weight.dtype==torch.bfloat16
  assert weight.shape==(12288,2048) and packed.shape==(12288,2048) and packed.stride()==(1,12288) and packed.T.is_contiguous() and scales.shape==(12288,)
  bank=sys.modules['_nn_silu_fx_NN_policy'].POINTERS[packed.data_ptr()];assert bank['bank_id']==bank_id
  capturing=torch.cuda.is_current_stream_capturing();y=torch.empty_like(x);rr=torch.empty_like(residual);q=torch.empty_like(x,dtype=torch.int8);a=torch.empty((64,),device=x.device,dtype=torch.float32);out=torch.empty((64,6144),device=x.device,dtype=x.dtype)
  nc=norm.run(x,residual,norm_weight,y,rr,q,a);gc=gu.fused(q,packed.T,a,scales,out,q.view(torch.int32),q.view(torch.bfloat16),config)
  trace=sys.modules.get('_tianshu_long_segments_actual_graph_trace');keygraph=getattr(trace,'STATE',{}).get('active_graph');key=str((bank_id,capturing,keygraph));row=record['routes'].setdefault(key,dict(bank_id=bank_id,prefix=bank['prefix'],M=64,K=2048,N=12288,actually_inside_capture=capturing,graph_key=keygraph,calls=0,standalone_A8_quantizer_executed=False,implementation='IR_RMS_A8_then_original_NN_two_dot_SiLU'))
  if 'norm_binary_sha256' not in row:
   nh=hashlib.sha256(nc.kernel).hexdigest();gh=hashlib.sha256(gc.kernel).hexdigest();assert nh==record['expected_norm_binary'] and gh==record['expected_GEMM_binary'];row.update(norm_binary_sha256=nh,norm_component_binary_exact=True,GEMM_binary_sha256=gh,GEMM_component_binary_exact=True,norm_cache_hash=nc.hash,GEMM_cache_hash=gc.hash)
  row['calls']+=1
  oldkey='actual_GEMM_from_RMS_A8_GU:'+key;old=nn.RECORD['routes'].setdefault(oldkey,dict(bank_id=bank_id,prefix=bank['prefix'],role='gateup',M=64,K=2048,N=12288,output_columns=6144,actually_inside_capture=capturing,graph_key=keygraph,implementation='NN_two_dot_fused_SiLU',configuration=config,kernel_source=record['GEMM_source'],kernel_sha256=record['GEMM_sha256'],calls=0,binary_sha256=row['GEMM_binary_sha256'],component_binary_exact_match=True,cache_hash=gc.hash,n_regs=gc.n_regs,n_spills=gc.n_spills,shared=gc.metadata.shared,wrapper_source=record['source'],wrapper_sha256=record['source_sha256'],standalone_A8_quantizer_executed=False,norm_and_A8_fused=True,scope='actual original GEMM kernel invocation; not original custom_op function counter'))
  old['calls']+=1
  if not capturing and bank_id==41 and row['calls']&(row['calls']-1)==0:save();nn.SAVE()
  return out,rr
 @fused.register_fake
 def fake(x,residual,norm_weight,weight,packed,scales,bank_id):return torch.empty((64,6144),device=x.device,dtype=x.dtype),torch.empty_like(residual)
 cc=cfg.compilation_config;previous=cc.inductor_compile_config['post_grad_custom_post_pass'];fx=load('_RMS_A8_GU_fx_pass',BASE/'fx_pass.py');cc.inductor_compile_config['post_grad_custom_post_pass']=fx.FusionPass(previous,mode,directory/'fx_pass')
 original_capture=ModelRunnerFL.capture_model
 @functools.wraps(original_capture)
 def capture(owner,*args,**kwargs):
  cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();banks=sys.modules['_nn_silu_fx_NN_policy'].LAYERS;sample=next(iter(banks.values()));x=torch.zeros((64,2048),device=sample.weight.device,dtype=torch.bfloat16);r=torch.zeros_like(x);w=torch.ones((2048,),device=x.device,dtype=x.dtype);y=torch.empty_like(x);rr=torch.empty_like(x);q=torch.empty_like(x,dtype=torch.int8);s=torch.empty((64,),device=x.device,dtype=torch.float32);c=norm.run(x,r,w,y,rr,q,s);torch.cuda.synchronize();assert hashlib.sha256(c.kernel).hexdigest()==record['expected_norm_binary'] and bool((y==0).all()) and bool((rr==0).all()) and bool((q==0).all());del x,r,w,y,rr,q,s
  assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state());record['warm_before_capture_RNG_unchanged']=True;save();value=original_capture(owner,*args,**kwargs)
  captured=[r for r in record['routes'].values() if r['actually_inside_capture']]
  if mode=='candidate':assert {r['bank_id'] for r in captured}==set(range(42)) and all(r['graph_key'] is not None and r['norm_component_binary_exact'] and r['GEMM_component_binary_exact'] for r in captured)
  else:assert not captured
  record.update(completed=True,all42_new_norm_capture_verified=(mode=='candidate'));save();nn.SAVE();return value
 ModelRunnerFL.capture_model=capture;atexit.register(save);save()
