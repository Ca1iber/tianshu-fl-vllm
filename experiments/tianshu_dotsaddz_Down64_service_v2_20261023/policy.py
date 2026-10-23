# SPDX-License-Identifier: Apache-2.0
import atexit,functools,hashlib,inspect,json,os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;COMPONENT=EXP/'tianshu_native_dotsaddz_down42_20261023'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def install(worker,directory):
 import numpy as np,torch
 from vllm.model_executor.layers.linear import UnquantizedLinearMethod
 from vllm_fl.worker.model_runner import ModelRunnerFL
 assert not getattr(ModelRunnerFL,'_tianshu_dots_Down64',False);ModelRunnerFL._tianshu_dots_Down64=True
 assert worker.cache_config.kv_cache_memory_bytes is None and worker.vllm_config.speculative_config is None
 proof=read(COMPONENT/'INDEPENDENT_CPU_GATE_32g.json');cpu=read(COMPONENT/'CPU_GATE_32g.json');selection=read(BASE/'SELECTED_BANKS_32g.json');assert proof['complete'] and proof['all_numeric_passed'] and proof['result_sha256']==sha(Path(proof['raw_directory'])/'result.json') and selection['CPU_gate_sha256']==sha(COMPONENT/'CPU_GATE_32g.json');expected={f"model.layers.{b['layer']}.mlp.down_proj":b for b in selection['banks']};assert len(expected)==42
 libraries={p:h for p,h in cpu['sources'].items() if Path(p).name in ('libtorch_cuda.so','libcuinfer.so.7','libcuinfer_blas.so')};assert len(libraries)==3 and all(sha(p)==h for p,h in libraries.items())
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);banks={};record=dict(complete=False,phase='loading',banks=[],routes={},warmups=[],profile_probes=[],profile_windows=[],source=str(BASE/'policy.py'),source_sha256=sha(BASE/'policy.py'),selection_source=str(BASE/'SELECTED_BANKS_32g.json'),selection_sha256=sha(BASE/'SELECTED_BANKS_32g.json'),component_gate=str(COMPONENT/'INDEPENDENT_CPU_GATE_32g.json'),component_gate_sha256=sha(COMPONENT/'INDEPENDENT_CPU_GATE_32g.json'),libraries=libraries,additional_packed_payload_bytes=0,original_checkpoint_not_modified=True,original_BF16_and_W8_retained=True,no_global_tensor_workspace=True,Python_calls_are_not_graph_replays=True)
 def save():
  p=directory/f'route_{os.getpid()}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(p)
 def thash(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 def profiler(function):
  frame=inspect.currentframe()
  try:
   while frame:
    if frame.f_code.co_name==function and (function!='load_model' or frame.f_globals.get('__name__')=='vllm_fl.worker.model_runner'):
     if function=='load_model' and 'm' in frame.f_locals:
      p=frame.f_locals['m'];assert hasattr(p,'initial_memory') and not hasattr(p,'final_memory');return dict(function=function,source=frame.f_code.co_filename,source_sha256=sha(frame.f_code.co_filename),initial_memory=int(p.initial_memory),active=True)
     if function=='determine_available_memory' and frame.f_globals.get('__name__')=='vllm_fl.worker.worker':return dict(function=function,source=frame.f_code.co_filename,source_sha256=sha(frame.f_code.co_filename),active='profile_result' in frame.f_locals,manual_KV_bytes=worker.cache_config.kv_cache_memory_bytes)
    frame=frame.f_back
   raise AssertionError('W4 work outside actual native memory profiler')
  finally:del frame
 def compute(x,bank):return torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,bank._tianshu_dots_w,128,bank._tianshu_dots_s,bank._tianshu_dots_z,size_n=2048)
 original_process=UnquantizedLinearMethod.process_weights_after_loading;original_apply=UnquantizedLinearMethod.apply
 @functools.wraps(original_process)
 def process(method,layer):
  result=original_process(method,layer)
  if tuple(layer.weight.shape)!=(2048,6144):return result
  prefix=layer.prefix;proof=expected[prefix];context=profiler('load_model');assert original_banks[int(layer._tianshu_native_down_bank)]['layer'] is layer;assert context['active'] and hasattr(layer,'_tianshu_native_down_bank');version=int(layer.weight._version);assert thash(layer.weight)==proof['original_sha256'];assert sha(proof['file'])==proof['file_sha256'];a=np.load(proof['file'],allow_pickle=False);entries=[]
  for key,name in [('W','w'),('S','s'),('Z','z')]:
   arr=a[key].copy();t=torch.from_numpy(arr);t=t.view(torch.bfloat16) if arr.dtype==np.uint16 else t;t=t.to(device=layer.weight.device).contiguous();layer.register_buffer('_tianshu_dots_'+name,t,persistent=False);expected_sha=proof[{'W':'pack_sha256','S':'scale_sha256','Z':'zero_sha256'}[key]];assert thash(t)==expected_sha;entries.append(dict(name=name,shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),sha256=expected_sha,bytes=t.numel()*t.element_size()))
  assert int(layer.weight._version)==version;bank_id=int(layer._tianshu_native_down_bank);assert bank_id not in banks;layer._tianshu_dots_bank=bank_id;banks[bank_id]=layer;payload=sum(v['bytes'] for v in entries);record['additional_packed_payload_bytes']+=payload;record['banks'].append(dict(bank_id=bank_id,prefix=prefix,layer=proof['layer'],original_sha256=proof['original_sha256'],packed_file=proof['file'],packed_file_sha256=proof['file_sha256'],entries=entries,payload_bytes=payload,inside_DeviceMemoryProfiler=context,original_weight_version_preserved=True));save()
  if len(banks)==42:
   before_cpu=torch.get_rng_state().clone();before_gpu=torch.cuda.get_rng_state().clone();zero=torch.zeros((64,6144),device=layer.weight.device,dtype=torch.bfloat16)
   for bid,bank in banks.items():
    o=compute(zero,bank);torch.cuda.synchronize();assert bool((o==0).all());record['warmups'].append(dict(bank_id=bid,M=64,zero_output=True,inside_DeviceMemoryProfiler=profiler('load_model')));del o
   del zero;assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_gpu,torch.cuda.get_rng_state());record.update(phase='before_profile',load_complete=True,load_RNG_unchanged=True);save()
  return result
 UnquantizedLinearMethod.process_weights_after_loading=process
 original_profile=ModelRunnerFL.profile_run
 @functools.wraps(original_profile)
 def profile(owner,*args,**kwargs):
  previous=record['phase'];record['phase']='profile';context=profiler('determine_available_memory');assert context['active'] and context['manual_KV_bytes'] is None;before_cpu=torch.get_rng_state().clone();before_gpu=torch.cuda.get_rng_state().clone();window=dict(context=context,allocated_before=int(torch.cuda.memory_allocated()),peak_before=int(torch.cuda.max_memory_allocated()))
  try:
   result=original_profile(owner,*args,**kwargs);before_cpu=torch.get_rng_state().clone();before_gpu=torch.cuda.get_rng_state().clone();bank0=banks[0];zero=torch.zeros((64,6144),device=bank0.weight.device,dtype=torch.bfloat16)
   for bid,bank in banks.items():
    o=compute(zero,bank);torch.cuda.synchronize();assert bool((o==0).all());record['profile_probes'].append(dict(bank_id=bid,M=64,zero_output=True,explicit_scratch_probe_not_main_M4096_path=True,inside_native_worker_memory_profile=profiler('determine_available_memory')));del o
   del zero;assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_gpu,torch.cuda.get_rng_state());window['RNG_unchanged']=True;return result
  finally:torch.cuda.synchronize();window.update(allocated_after=int(torch.cuda.memory_allocated()),peak_after=int(torch.cuda.max_memory_allocated()));record['profile_windows'].append(window);record['phase']=previous;save()
 ModelRunnerFL.profile_run=profile
 original_capture=ModelRunnerFL.capture_model
 @functools.wraps(original_capture)
 def capture(owner,*args,**kwargs):
  record['phase']='capture';save()
  try:r=original_capture(owner,*args,**kwargs)
  except BaseException:save();raise
  storages={}
  for t in list(owner.model.parameters())+list(owner.model.buffers()):
   if t.is_cuda:storage=t.untyped_storage();storages[(str(t.device),storage.data_ptr())]=storage.nbytes()
  total=sum(storages.values());assert int(owner.model_memory_usage)+16*1024**2>=total;assert len(banks)==42 and len(record['warmups'])==42 and len(record['profile_probes'])==42;record.update(phase='runtime',complete=True,capture_completed=True,model_memory_usage_bytes=int(owner.model_memory_usage),all_model_unique_parameter_buffer_storage_bytes=total,own_buffers_in_actual_model_accounting=True);save();return r
 ModelRunnerFL.capture_model=capture;atexit.register(save);save()
 # Replace the kernel of the actual opaque op already present in the traced graph.
 from torch._library.custom_ops import OPDEFS
 definition=OPDEFS['tianshu_native_down::linear'];original_backend=definition._backend_fns['cuda'];original_fn=definition._init_fn;original_source=str(Path(inspect.getsourcefile(original_fn)).resolve());assert original_source==str(EXP/'tianshu_native_int8_down_service_v2_20261022/native_down_policy.py');assert 'cuda' not in definition._disabled_kernel
 cells=dict(zip(original_fn.__code__.co_freevars,original_fn.__closure__));original_banks=cells['banks'].cell_contents;assert isinstance(original_banks,dict) and not original_banks
 record.update(actual_original_op_name='tianshu_native_down::linear',original_op_schema=str(definition._opoverload._schema),original_init_source=original_source,original_init_sha256=sha(original_source),original_init_freevars=list(original_fn.__code__.co_freevars),genuine_original_bank_closure_bound=True,public_register_kernel_used=True,original_CUDA_backend_retained_for_other_M=True,new_UnquantizedLinearMethod_apply_override=False)
 def routed(x,weight,packed,scales,bank_id):
  if x.ndim!=2 or x.shape[0]!=64 or x.dtype!=torch.bfloat16 or not x.is_contiguous():return original_backend(x,weight,packed,scales,bank_id)
  assert len(banks)==42 and original_banks[bank_id]['layer'] is banks[bank_id];bank=banks[bank_id];assert weight.data_ptr()==bank.weight.data_ptr() and packed.data_ptr()==bank._tianshu_native_down_packed.data_ptr() and scales.data_ptr()==bank._tianshu_native_down_scale.data_ptr();capturing=torch.cuda.is_current_stream_capturing();key=str((bank_id,record['phase'],capturing));trace=getattr(sys.modules.get('_tianshu_long_segments_actual_graph_trace'),'STATE',{});row=record['routes'].setdefault(key,dict(bank_id=bank_id,prefix=bank.prefix,M=64,K=6144,N=2048,phase=record['phase'],capturing=capturing,graph_key=trace.get('active_graph'),input_stride=list(x.stride()),implementation='native_W4A16_dotsaddz',original_op_schema_preserved=True,actual_parent_bank_identity_matched=True,calls=0));row['calls']+=1;out=compute(x,bank)
  if bank_id==41 and row['calls']&(row['calls']-1)==0:save()
  return out
 definition.register_kernel('cuda',routed);record['CUDA_route_registered']=True;save()
