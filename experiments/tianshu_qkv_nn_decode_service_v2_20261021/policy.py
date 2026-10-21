# SPDX-License-Identifier: Apache-2.0
"""QKV M64 NN integer projection, enabled during real graph capture."""
import atexit,functools,hashlib,importlib.util,inspect,json,os,re,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
PACK=EXP/'tianshu_qkv_prefill_int8_service_v2_20261021/qkv_prefill_weight_pack.py'
SIGNATURE=EXP/'tianshu_qkv_prefill_int8_service_v2_20261021/CHECKPOINT_SIGNATURES_32g.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def install(worker,mode,directory):
 assert mode=='candidate'
 import torch
 from vllm.model_executor.layers.linear import UnquantizedLinearMethod
 from vllm_fl.worker.model_runner import ModelRunnerFL
 cfg=worker.vllm_config;hf=cfg.model_config.hf_config
 assert cfg.model_config.model=='/workspace/MiniCPM5-2B' and cfg.model_config.quantization is None and cfg.speculative_config is None and cfg.lora_config is None
 assert (hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads)==(2048,42,16,2)
 assert cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.async_scheduling
 assert cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==cfg.parallel_config.data_parallel_size==1
 selection=json.loads((BASE/'selected_plan.json').read_text());component=json.loads(Path(selection['component_gate']).read_text());kernel_path=Path(selection['kernel_source'])
 assert component['complete'] and component['service_candidate_worth_testing'] and component['all_saved_INT32_elements_independently_CPU_recomputed']==9400320
 assert sha(selection['component_gate'])==selection['component_gate_sha256'] and sha(kernel_path)==selection['kernel_sha256']
 assert all(sha(p)==h for p,h in component['frozen_sources'].items())
 pack=load('_nn_QKV_checkpoint_pack',PACK);kernel=load('_nn_QKV_integer_kernel',kernel_path);manifest=json.loads(SIGNATURE.read_text());assert manifest['complete'] and len(manifest['banks'])==42
 expected={r['prefix']:r for r in manifest['banks']};config=selection['configuration']
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
 original_process=UnquantizedLinearMethod.process_weights_after_loading;original_apply=UnquantizedLinearMethod.apply;original_capture=ModelRunnerFL.capture_model
 banks={};dummy=None
 record=dict(installed=True,mode=mode,completed=False,stage='loading',banks=[],routes={},warmups=[],new_integer_M=[64],other_M_delegates_original_method=True,original_BF16_weights_retained=True,packing_inside_actual_active_DeviceMemoryProfiler=True,new_path_enabled_during_capture=True,no_temporary_tensor_global_cache=True,Python_call_counts_are_not_Graph_replay_counts=True,policy_source=str(BASE/'policy.py'),policy_sha256=sha(__file__),kernel_source=str(kernel_path),kernel_sha256=sha(kernel_path),selection_source=str(BASE/'selected_plan.json'),selection_sha256=sha(BASE/'selected_plan.json'),checkpoint_signature_source=str(SIGNATURE),checkpoint_signature_sha256=sha(SIGNATURE),configuration=config,expected_component_binary_sha256=selection['expected_component_binary_sha256'],original_apply_source=str(Path(inspect.getsourcefile(original_apply)).resolve()))
 def save():
  p=directory/f'route_{os.getpid()}.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(record,indent=2)+'\n');t.replace(p)
 def emit(row):
  row['calls']+=1
  if row['prefix']=='model.layers.41.self_attn.qkv_proj' and row['calls']&(row['calls']-1)==0:save()
 def meta(compiled):
  binary=hashlib.sha256(compiled.kernel).hexdigest();assert binary==selection['expected_component_binary_sha256']
  return dict(kernel_sha256=selection['kernel_sha256'],binary_sha256=binary,component_binary_exact_match=True,cache_hash=compiled.hash,n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared)
 def profiler():
  frame=inspect.currentframe().f_back
  try:
   while frame:
    if frame.f_globals.get('__name__')=='vllm_fl.worker.model_runner' and frame.f_code.co_name=='load_model' and 'm' in frame.f_locals:
     p=frame.f_locals['m'];assert hasattr(p,'initial_memory') and not hasattr(p,'final_memory');return dict(active=True,source=str(Path(frame.f_code.co_filename).resolve()),sha256=sha(frame.f_code.co_filename),initial_memory=int(p.initial_memory))
    frame=frame.f_back
   raise AssertionError('QKV packing outside active native DeviceMemoryProfiler')
  finally:del frame
 def warm(layer):
  before_cpu,before_gpu=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone();threads=torch.get_num_threads()
  try:
   torch.set_num_threads(4)
   with torch.inference_mode():
    cpu_a=((((torch.arange(64,dtype=torch.int32)[:,None]%17)-8)*((torch.arange(2048,dtype=torch.int32)[None,:]%13)-6))/8.).to(torch.bfloat16);cpu_a[0].zero_();a=cpu_a.to(layer.weight.device);q=torch.empty((64,2048),device=a.device,dtype=torch.int8);s=torch.empty(64,device=a.device,dtype=torch.float32);out=torch.empty((64,2560),device=a.device,dtype=torch.bfloat16)
    for _ in range(2):kernel.quantize(a,q,s);compiled=kernel.mm(q,layer._tianshu_qkv_nn_packed,s,layer._tianshu_qkv_nn_scales,out,dummy,config)
    torch.cuda.synchronize();metadata=meta(compiled);v=cpu_a.float();asc=v.abs().amax(1).clamp_min(1.e-8)/127.;aq=(v/asc[:,None]).round().clamp(-127,127).to(torch.int8);assert torch.equal(q.cpu(),aq) and torch.equal(s.cpu(),asc)
    rows=torch.tensor([0,1,32,63],device='cpu',dtype=torch.int64);cols=torch.arange(32,device='cpu',dtype=torch.int64)*2559//31;w=layer._tianshu_qkv_nn_packed.detach().cpu().T.contiguous()[cols];ws=layer._tianshu_qkv_nn_scales.detach().cpu()[cols];integer=aq[rows].to(torch.int64)@w.to(torch.int64).T;expected_y=integer.float()*asc[rows,None]*ws[None,:];actual=out[rows.to(a.device)[:,None],cols.to(a.device)[None,:]].cpu()
    assert torch.isfinite(actual).all() and ((actual.float()-expected_y).abs()<=.02+.01*expected_y.abs()).all()
    record['warmups'].append(dict(M=64,prefix=layer.prefix,all_A_quantization_exact=True,selected_real_W_INT64_outputs_passed=True,CPU_selected_elements=actual.numel(),**metadata))
  finally:torch.set_num_threads(threads)
  assert torch.equal(torch.get_rng_state(),before_cpu) and torch.equal(torch.cuda.get_rng_state(),before_gpu)
  record.update(warm_RNG_unchanged=True,warm_CPU_threads_restored=True,warm_before_actual_graph_capture=True)
 @functools.wraps(original_process)
 def process(method,layer):
  nonlocal dummy
  value=original_process(method,layer);prefix=getattr(layer,'prefix','')
  if not re.fullmatch(r'model\.layers\.\d+\.self_attn\.qkv_proj',prefix):return value
  assert tuple(layer.weight.shape)==(2560,2048) and layer.weight.dtype==torch.bfloat16 and layer.weight.is_contiguous()
  i=int(prefix.split('.')[2]);assert i not in banks;proof=profiler();weight=layer.weight;version=int(weight._version);pointer=weight.data_ptr();started=time.perf_counter();cpu=weight.detach().cpu().contiguous();signature=expected[prefix]
  assert pack.tensor_sha(cpu)==signature['original_BF16_sha256'];q,s=pack.pack(cpu)
  assert pack.tensor_sha(q)==signature['packed_logical_NK_sha256'] and pack.tensor_sha(s)==signature['scales_sha256']
  packed=q.T.contiguous().to(weight.device);scales=s.to(weight.device)
  assert pack.tensor_sha(packed.detach().cpu())==signature['packed_physical_KN_sha256'] and pack.tensor_sha(scales.detach().cpu())==signature['scales_sha256']
  assert weight.data_ptr()==pointer and int(weight._version)==version and pack.tensor_sha(weight.detach().cpu())==signature['original_BF16_sha256']
  layer._tianshu_qkv_nn_bank=i;layer._tianshu_qkv_nn_packed=packed;layer._tianshu_qkv_nn_scales=scales;banks[i]=dict(layer=layer,method=method,prefix=prefix)
  if dummy is None:dummy=torch.empty(1,device=weight.device,dtype=torch.int32)
  record['banks'].append(dict(bank_id=i,prefix=prefix,signature=signature,payload_bytes=packed.numel()+scales.numel()*4,inside_profiler=proof,original_weight_unchanged=True,physical_shape=list(packed.shape),physical_stride=list(packed.stride()),packing_seconds=time.perf_counter()-started))
  if i==0:warm(layer)
  save();return value
 UnquantizedLinearMethod.process_weights_after_loading=process
 @torch.library.custom_op('tianshu_qkv_NN_decode::linear',mutates_args=(),device_types='cuda')
 def linear(x:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
  bank=banks[bank_id];m=x.shape[0];capturing=torch.cuda.is_current_stream_capturing();use=(m==64 and x.is_contiguous());implementation='NN_A8W8_M64' if use else 'delegate_original_method'
  trace=getattr(sys.modules.get('_tianshu_long_segments_actual_graph_trace'),'STATE',{});graph_key=trace.get('active_graph');key=str((bank_id,m,record['stage'],capturing,graph_key,implementation))
  row=record['routes'].setdefault(key,dict(prefix=bank['prefix'],M=m,stage=record['stage'],actually_inside_capture=capturing,graph_key=graph_key,implementation=implementation,input_stride=list(x.stride()),calls=0))
  if not use:
   value=original_apply(bank['method'],bank['layer'],x,None);emit(row);return value
  assert packed.shape==(2048,2560) and packed.is_contiguous() and scales.shape==(2560,)
  q=torch.empty((64,2048),device=x.device,dtype=torch.int8);s=torch.empty(64,device=x.device,dtype=torch.float32);out=torch.empty((64,2560),device=x.device,dtype=torch.bfloat16)
  kernel.quantize(x,q,s);compiled=kernel.mm(q,packed,s,scales,out,dummy,config)
  if 'binary_sha256' not in row:row.update(configuration=config,**meta(compiled))
  emit(row);return out
 @linear.register_fake
 def fake(x,weight,packed,scales,bank_id):return torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=x.dtype)
 @functools.wraps(original_apply)
 def apply(method,layer,x,bias=None):
  if bias is not None or not hasattr(layer,'_tianshu_qkv_nn_bank') or x.ndim!=2 or x.dtype!=torch.bfloat16 or x.stride(-1)!=1:return original_apply(method,layer,x,bias)
  return linear(x,layer.weight,layer._tianshu_qkv_nn_packed,layer._tianshu_qkv_nn_scales,int(layer._tianshu_qkv_nn_bank))
 UnquantizedLinearMethod.apply=apply
 @functools.wraps(original_capture)
 def capture(owner,*args,**kwargs):
  assert set(banks)==set(range(42)) and record.get('warm_before_actual_graph_capture') and not record['completed'];record['stage']='capturing';save();answer=original_capture(owner,*args,**kwargs)
  captured=[r for r in record['routes'].values() if r['M']==64 and r['actually_inside_capture'] and r['implementation']=='NN_A8W8_M64'];assert len({r['prefix'] for r in captured})==42 and all(r['component_binary_exact_match'] for r in captured)
  payload=sum(r['payload_bytes'] for r in record['banks']);assert payload==220631040
  record.update(stage='active',completed=True,all_42_M64_real_capture_routes_verified=True,actual_extra_packed_payload_bytes=payload,actual_model_memory_usage_bytes=int(owner.model_memory_usage),original_471_model_operators_remain_loaded=True);save();return answer
 ModelRunnerFL.capture_model=capture;atexit.register(save);save()
