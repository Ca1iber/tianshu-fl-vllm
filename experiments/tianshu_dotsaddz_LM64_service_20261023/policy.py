# SPDX-License-Identifier: Apache-2.0
"""DRAFT: actual LM bank / M64 only; do not execute before frozen CPU gates."""
import atexit,functools,hashlib,inspect,json,os
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;COMPONENT=EXP/'tianshu_native_W4_lm64_20261023'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def install(worker,directory):
 import numpy as np,torch
 from torch._library.custom_ops import OPDEFS
 from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead,UnquantizedEmbeddingMethod
 from vllm_fl.worker.model_runner import ModelRunnerFL
 assert read(BASE/'CPU_GATE_32g.json')['complete']
 assert not getattr(ModelRunnerFL,'_tianshu_dots_LM64',False);ModelRunnerFL._tianshu_dots_LM64=True
 assert worker.cache_config.kv_cache_memory_bytes is None and worker.vllm_config.speculative_config is None
 proof=read(COMPONENT/'INDEPENDENT_CPU_GATE_32g.json');selection=read(BASE/'SELECTED_LM_32g.json')
 assert proof['complete'] and proof['result_sha256']==sha(Path(proof['raw_directory'])/'result.json') and selection['CPU_gate_sha256']==sha(COMPONENT/'CPU_GATE_32g.json')
 definition=OPDEFS['tianshu_dense_w8a8::linear'];original_backend=definition._backend_fns['cuda'];fn=definition._init_fn;source=str(Path(inspect.getsourcefile(fn)).resolve())
 assert source==str(EXP/'tianshu_lm_precision_service_20261020/policy.py') and 'cuda' not in definition._disabled_kernel
 cells=dict(zip(fn.__code__.co_freevars,fn.__closure__));original_banks=cells['banks'].cell_contents;assert isinstance(original_banks,dict) and not original_banks
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);bank=None
 record=dict(complete=False,phase='loading',routes={},banks=[],warmups=[],profile_probes=[],profile_windows=[],source=str(BASE/'policy.py'),source_sha256=sha(BASE/'policy.py'),selection_source=str(BASE/'SELECTED_LM_32g.json'),selection_sha256=sha(BASE/'SELECTED_LM_32g.json'),additional_packed_payload_bytes=0,actual_original_op_name='tianshu_dense_w8a8::linear',original_op_schema=str(definition._opoverload._schema),original_init_source=source,original_init_sha256=sha(source),original_init_freevars=list(fn.__code__.co_freevars),genuine_original_bank_closure_bound=True,public_register_kernel_used=True,original_CUDA_backend_retained_for_other_M_and_GU=True,no_new_apply_override=True,original_checkpoint_and_BF16_W8_retained=True,Python_calls_are_not_Graph_replays=True,no_global_tensor_workspace=True)
 def save():
  p=directory/f'route_{os.getpid()}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(p)
 def thash(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 def context(function):
  frame=inspect.currentframe()
  try:
   while frame:
    if frame.f_code.co_name==function:
     if function=='load_model' and frame.f_globals.get('__name__')=='vllm_fl.worker.model_runner' and 'm' in frame.f_locals:
      p=frame.f_locals['m'];assert hasattr(p,'initial_memory') and not hasattr(p,'final_memory');return dict(function=function,source=frame.f_code.co_filename,source_sha256=sha(frame.f_code.co_filename),initial_memory=int(p.initial_memory),active=True)
     if function=='determine_available_memory' and frame.f_globals.get('__name__')=='vllm_fl.worker.worker':return dict(function=function,source=frame.f_code.co_filename,source_sha256=sha(frame.f_code.co_filename),active='profile_result' in frame.f_locals,manual_KV_bytes=worker.cache_config.kv_cache_memory_bytes)
    frame=frame.f_back
   raise AssertionError('LM64 W4 outside actual native memory profiler')
  finally:del frame
 def compute(x):return torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,bank._tianshu_lm4_w,128,bank._tianshu_lm4_s,bank._tianshu_lm4_z,size_n=130560)
 original_process=UnquantizedEmbeddingMethod.process_weights_after_loading
 @functools.wraps(original_process)
 def process(method,layer):
  nonlocal bank
  result=original_process(method,layer)
  if not isinstance(layer,ParallelLMHead):return result
  assert bank is None and tuple(layer.weight.shape)==(130560,2048);bid=int(layer._tianshu_dense_w8_bank);assert original_banks[bid]['role']=='lm_head' and original_banks[bid]['layer'] is layer
  evidence=context('load_model');assert evidence['active'];version=int(layer.weight._version);assert thash(layer.weight)==selection['W8_signature']['original_BF16_sha256'] and thash(layer._tianshu_dense_w8_packed)==selection['W8_signature']['packed_sha256'] and thash(layer._tianshu_dense_w8_scale)==selection['W8_signature']['scales_sha256']
  entries=[]
  for name,item in selection['buffers'].items():
   assert sha(item['file'])==item['file_sha256'];a=np.load(item['file'],allow_pickle=False).copy();assert hashlib.sha256(a.tobytes()).hexdigest()==item['array_sha256'];t=torch.from_numpy(a);t=t.view(torch.bfloat16) if a.dtype==np.uint16 else t;t=t.to(layer.weight.device).contiguous();assert thash(t)==item['array_sha256'];layer.register_buffer('_tianshu_lm4_'+name,t,persistent=False);entries.append(dict(name=name,shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),sha256=item['array_sha256'],bytes=t.numel()*t.element_size()))
  bank=layer;assert int(layer.weight._version)==version;record['banks'].append(dict(bank_id=bid,original_BF16_sha256=selection['W8_signature']['original_BF16_sha256'],entries=entries,inside_DeviceMemoryProfiler=evidence,original_weight_version_preserved=True));record['additional_packed_payload_bytes']=sum(v['bytes'] for v in entries)
  cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();zero=torch.zeros((64,2048),device=layer.weight.device,dtype=torch.bfloat16);out=compute(zero);torch.cuda.synchronize();assert bool((out==0).all());del out,zero
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());record['warmups'].append(dict(bank_id=bid,M=64,zero_output=True,inside_DeviceMemoryProfiler=context('load_model')));record.update(load_complete=True,load_RNG_unchanged=True,phase='before_profile');save();return result
 UnquantizedEmbeddingMethod.process_weights_after_loading=process
 original_profile=ModelRunnerFL.profile_run
 @functools.wraps(original_profile)
 def profile(owner,*args,**kwargs):
  previous=record['phase'];record['phase']='profile';evidence=context('determine_available_memory');assert evidence['active'] and evidence['manual_KV_bytes'] is None;window=dict(context=evidence,allocated_before=int(torch.cuda.memory_allocated()),peak_before=int(torch.cuda.max_memory_allocated()))
  try:
   result=original_profile(owner,*args,**kwargs);cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();zero=torch.zeros((64,2048),device=bank.weight.device,dtype=torch.bfloat16);out=compute(zero);torch.cuda.synchronize();assert bool((out==0).all());del out,zero
   assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());record['profile_probes'].append(dict(M=64,zero_output=True,explicit_scratch_probe=True,inside_native_worker_memory_profile=context('determine_available_memory')));window['RNG_unchanged']=True;return result
  finally:torch.cuda.synchronize();window.update(allocated_after=int(torch.cuda.memory_allocated()),peak_after=int(torch.cuda.max_memory_allocated()));record['profile_windows'].append(window);record['phase']=previous;save()
 ModelRunnerFL.profile_run=profile
 original_capture=ModelRunnerFL.capture_model
 @functools.wraps(original_capture)
 def capture(owner,*args,**kwargs):
  record['phase']='capture';save()
  try:result=original_capture(owner,*args,**kwargs)
  except BaseException:save();raise
  storages={}
  for t in list(owner.model.parameters())+list(owner.model.buffers()):
   if t.is_cuda:s=t.untyped_storage();storages[(str(t.device),s.data_ptr())]=s.nbytes()
  total=sum(storages.values());assert int(owner.model_memory_usage)+16*1024**2>=total and len(record['banks'])==len(record['warmups'])==len(record['profile_probes'])==1;record.update(phase='runtime',complete=True,capture_completed=True,model_memory_usage_bytes=int(owner.model_memory_usage),all_model_unique_parameter_buffer_storage_bytes=total,own_buffers_in_actual_model_accounting=True);save();return result
 ModelRunnerFL.capture_model=capture
 def routed(x,weight,packed,scales,bank_id):
  actual=original_banks[bank_id]
  if actual['role']!='lm_head' or x.ndim!=2 or x.shape[0]!=64 or x.dtype!=torch.bfloat16 or not x.is_contiguous():return original_backend(x,weight,packed,scales,bank_id)
  assert bank is not None and actual['layer'] is bank and weight.data_ptr()==bank.weight.data_ptr() and packed.data_ptr()==bank._tianshu_dense_w8_packed.data_ptr() and scales.data_ptr()==bank._tianshu_dense_w8_scale.data_ptr();capturing=torch.cuda.is_current_stream_capturing();key=str((record['phase'],capturing));row=record['routes'].setdefault(key,dict(bank_id=bank_id,M=64,K=2048,N=130560,phase=record['phase'],capturing=capturing,input_stride=list(x.stride()),implementation='native_W4A16_dotsaddz',actual_parent_bank_identity_matched=True,calls=0));row['calls']+=1;out=compute(x)
  if row['calls']&(row['calls']-1)==0:save()
  return out
 definition.register_kernel('cuda',routed);record['CUDA_route_registered']=True;atexit.register(save);save()
