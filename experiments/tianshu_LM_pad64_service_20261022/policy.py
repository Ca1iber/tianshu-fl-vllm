# SPDX-License-Identifier: Apache-2.0
import atexit,functools,hashlib,importlib.util,inspect,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install(worker,mode,directory):
 import torch
 from vllm.model_executor.layers.logits_processor import LogitsProcessor
 from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
 from vllm_fl.worker.model_runner import ModelRunnerFL
 selection=json.loads((BASE/'selected_plan.json').read_text());minimum=selection['minimum_M'];directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);spec=importlib.util.spec_from_file_location('_pad64_new',EXP/'tianshu_LM_pad64_20261022/kernel.py');kernel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernel;spec.loader.exec_module(kernel);config=selection['configuration'];bank=None;compiled=None
 record=dict(installed=True,completed=False,mode=mode,phase='load',routes={},fallback_calls={},source=str(Path(__file__).resolve()),source_sha256=sha(__file__),kernel_source=str(EXP/'tianshu_LM_pad64_20261022/kernel.py'),kernel_sha256=sha(EXP/'tianshu_LM_pad64_20261022/kernel.py'),selection_source=str(BASE/'selected_plan.json'),selection_sha256=sha(BASE/'selected_plan.json'),minimum_M=minimum,maximum_M=64,physical_M=64,original_BF16_retained=True,no_new_weight_layout_or_payload=True,no_global_tensor_workspace=True,all_scratch_call_local=True,FP32_output_after_explicit_BF16_round=True,warmups=[],profile_probes=[],Python_calls_not_graph_replays=True)
 def save():
  p=directory/f'route_{__import__("os").getpid()}.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(record,indent=2)+'\n');t.replace(p)
 def th(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 def profiler():
  f=inspect.currentframe().f_back
  try:
   while f:
    if f.f_globals.get('__name__')=='vllm_fl.worker.model_runner' and f.f_code.co_name=='load_model' and 'm' in f.f_locals:
     p=f.f_locals['m'];assert hasattr(p,'initial_memory') and not hasattr(p,'final_memory');return dict(active=True,source=str(Path(f.f_code.co_filename).resolve()),sha256=sha(f.f_code.co_filename))
    f=f.f_back
   raise AssertionError('LM warm outside actual loader profiler')
  finally:del f
 def compute(x,packed,scales):
  nonlocal compiled
  m=x.shape[0];assert minimum<=m<=64 and x.shape[1]==2048 and x.dtype==torch.bfloat16 and x.stride(1)==1; q=torch.empty((64,2048),device=x.device,dtype=torch.int8);s=torch.empty(64,device=x.device,dtype=torch.float32);out=torch.empty((64,130560),device=x.device,dtype=torch.float32);dummy=torch.empty(1,device=x.device,dtype=torch.int32);qkernel=kernel.quantize_padded(x,q,s);gemm=kernel.mm(q,packed,s,scales,out,dummy,config)
  if compiled is None:
   compiled={name:dict(binary_sha256=hashlib.sha256(k.kernel).hexdigest(),cache_hash=k.hash,n_regs=k.n_regs,n_spills=k.n_spills,shared=k.metadata.shared) for name,k in (('quantizer',qkernel),('GEMM',gemm))};assert compiled==selection['expected_compiled'];record['compiled']=compiled
  return out[:m]
 @torch.library.custom_op('tianshu_lm_pad64::linear',mutates_args=(),device_types='cuda')
 def linear(x:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor)->torch.Tensor:
  m=x.shape[0];capturing=torch.cuda.is_current_stream_capturing();out=compute(x,packed,scales);key=str((m,capturing,record['phase']));r=record['routes'].setdefault(key,dict(M=m,physical_M=64,phase=record['phase'],capturing=capturing,calls=0,output_dtype='float32',source_sha256=record['source_sha256'],kernel_sha256=record['kernel_sha256']));r['calls']+=1
  if r['calls']&(r['calls']-1)==0:save()
  return out
 @linear.register_fake
 def fake(x,packed,scales):return torch.empty((x.shape[0],130560),device=x.device,dtype=torch.float32)
 original_process=UnquantizedEmbeddingMethod.process_weights_after_loading
 @functools.wraps(original_process)
 def process(method,layer):
  nonlocal bank
  result=original_process(method,layer)
  if tuple(layer.weight.shape)!=(130560,2048) or not hasattr(layer,'_tianshu_dense_w8_packed'):return result
  assert bank is None;bank=layer;context=profiler();version=int(layer.weight._version);cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();packed=layer._tianshu_dense_w8_packed;scales=layer._tianshu_dense_w8_scale;proof=next(p for p in json.loads((EXP/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text()) if p['prefix']=='lm_head');record['bank']=dict(original_BF16_sha256=th(layer.weight),packed_sha256=th(packed),scales_sha256=th(scales),inside_DeviceMemoryProfiler=context,packed_shape=list(packed.shape),packed_stride=list(packed.stride()),original_version=version);assert all(record['bank'][k]==proof[k] for k in ('original_BF16_sha256','packed_sha256','scales_sha256'));x=torch.zeros((64,2048),dtype=torch.bfloat16,device=layer.weight.device);out=compute(x,packed,scales);torch.cuda.synchronize();assert out.dtype==torch.float32 and bool((out==0).all()) and int(layer.weight._version)==version and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());record['warmups'].append(dict(M=64,zero_output=True,inside_DeviceMemoryProfiler=context,RNG_unchanged=True,FP32_output_bytes=64*130560*4));save();return result
 UnquantizedEmbeddingMethod.process_weights_after_loading=process
 original_logits=LogitsProcessor._get_logits;record['original_logits_source']=str(Path(inspect.getsourcefile(original_logits)).resolve());record['original_logits_sha256']=sha(record['original_logits_source'])
 @functools.wraps(original_logits)
 def get_logits(owner,hidden_states,lm_head,embedding_bias=None):
  m=hidden_states.shape[0]
  if mode=='candidate' and bank is lm_head and embedding_bias is None and owner.scale==1. and owner.soft_cap is None and minimum<=m<=64 and hidden_states.ndim==2 and hidden_states.dtype==torch.bfloat16 and hidden_states.stride(1)==1:
   logits=linear(hidden_states,lm_head._tianshu_dense_w8_packed,lm_head._tianshu_dense_w8_scale);logits=owner._gather_logits(logits);return None if logits is None else logits[...,:owner.org_vocab_size]
  record['fallback_calls'][str(m)]=record['fallback_calls'].get(str(m),0)+1
  return original_logits(owner,hidden_states,lm_head,embedding_bias)
 LogitsProcessor._get_logits=get_logits
 original_profile=ModelRunnerFL.profile_run
 @functools.wraps(original_profile)
 def profile(owner,*args,**kwargs):
  assert bank is not None;previous=record['phase'];record['phase']='profile';frame=inspect.currentframe().f_back;context=None
  try:
   while frame:
    if frame.f_globals.get('__name__')=='vllm_fl.worker.worker' and frame.f_code.co_name=='determine_available_memory':context=dict(active=('profile_result' in frame.f_locals),source=str(Path(frame.f_code.co_filename).resolve()),sha256=sha(frame.f_code.co_filename),manual_KV_bytes=worker.cache_config.kv_cache_memory_bytes);break
    frame=frame.f_back
  finally:del frame
  assert context and context['active'] and not context['manual_KV_bytes'];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();x=torch.zeros((64,2048),device=bank.weight.device,dtype=torch.bfloat16);out=compute(x,bank._tianshu_dense_w8_packed,bank._tianshu_dense_w8_scale);torch.cuda.synchronize();assert bool((out==0).all()) and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());record['profile_probes'].append(dict(explicit_zero_probe_not_real_request=True,M=64,FP32_output_bytes=out.numel()*4,context=context,peak_after=int(torch.cuda.max_memory_allocated()),zero_output=True,RNG_unchanged=True));del x,out
  try:return original_profile(owner,*args,**kwargs)
  finally:record['phase']=previous;save()
 ModelRunnerFL.profile_run=profile
 original_capture=ModelRunnerFL.capture_model
 @functools.wraps(original_capture)
 def capture(owner,*args,**kwargs):
  record['phase']='capture'
  try:result=original_capture(owner,*args,**kwargs)
  except BaseException:save();raise
  record.update(phase='runtime',completed=True,model_memory_usage_bytes=int(owner.model_memory_usage));save();return result
 ModelRunnerFL.capture_model=capture;atexit.register(save);save()
