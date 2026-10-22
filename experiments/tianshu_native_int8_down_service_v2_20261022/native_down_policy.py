# SPDX-License-Identifier: Apache-2.0
import atexit,functools,hashlib,importlib.util,inspect,json,os,re,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def install(worker,mode,scope,directory):
 import torch
 from vllm.model_executor.layers.linear import UnquantizedLinearMethod
 from vllm_fl.worker.model_runner import ModelRunnerFL
 assert mode in ('control','candidate') and scope in ('all','prefill') and worker.vllm_config.speculative_config is None
 assert worker.vllm_config.model_config.model=='/workspace/MiniCPM5-2B' and worker.vllm_config.scheduler_config.max_num_batched_tokens==4096 and worker.vllm_config.scheduler_config.max_num_seqs==64
 assert worker.cache_config.kv_cache_memory_bytes is None
 selected=json.loads((BASE/'selected_plan.json').read_text());assert selected['complete'] and selected['quality_not_yet_established']
 quant_path=EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py';epi_path=EXP/'tianshu_native_int8_lm64_20261022/kernel.py';pack_path=EXP/'tianshu_w8_gpu_packing_20261020/packing.py';dense_path=EXP/'tianshu_dense_w8a8_20261020/kernel.py'
 quant=load('_native_down_live_quant',quant_path);epilogue=load('_native_down_live_epilogue',epi_path);packer=load('_native_down_live_packer',pack_path);dense=load('_native_down_live_dense',dense_path)
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);banks={};expected={r['prefix']:r for r in selected['banks']};active=False;warmed=False
 record=dict(installed=True,mode=mode,scope=scope,supported_rows=[64,128,512,4096] if scope=='all' else [128,512,4096],phase='initializing',completed=False,capture_completed=False,banks=[],routes={},BF16_vendor_modules={},warmups=[],native_kernels={},profile_windows=[],source=str(Path(__file__).resolve()),source_sha256=sha(__file__),quantizer_source=str(quant_path),quantizer_sha256=sha(quant_path),epilogue_source=str(epi_path),epilogue_sha256=sha(epi_path),packer_source=str(pack_path),packer_sha256=sha(pack_path),dense_source=str(dense_path),dense_sha256=sha(dense_path),selection_source=str(BASE/'selected_plan.json'),selection_sha256=sha(BASE/'selected_plan.json'),packed_payload_bytes=0,one_new_Down_NN_layout_only=True,original_BF16_retained=True,no_global_tensor_workspace=True,Python_calls_are_not_Graph_replays=True)
 def save():
  path=directory/f'route_{os.getpid()}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
 def thash(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 def active_profiler():
  frame=inspect.currentframe()
  try:
   while frame:
    if frame.f_globals.get('__name__')=='vllm_fl.worker.model_runner' and frame.f_code.co_name=='load_model' and 'm' in frame.f_locals:
     profiler=frame.f_locals['m'];assert hasattr(profiler,'initial_memory') and not hasattr(profiler,'final_memory')
     return dict(active=True,source=str(Path(frame.f_code.co_filename).resolve()),sha256=sha(frame.f_code.co_filename))
    frame=frame.f_back
   raise AssertionError('Down packing/warmup outside actual DeviceMemoryProfiler')
  finally:del frame
 def compute(x,packed,scales):
  m=x.shape[0];block=selected['block_by_M'][str(m)];q=torch.empty((m,6144),device=x.device,dtype=torch.int8);asc=torch.empty(m,device=x.device,dtype=torch.float32);integer=torch.empty((m,2048),device=x.device,dtype=torch.int32);out=torch.empty((m,2048),device=x.device,dtype=torch.bfloat16)
  quant.quantize(x,q,asc);torch._int_mm(q,packed,out=integer);compiled=epilogue._dequant[((m*2048+block-1)//block,)](integer,asc,scales,out,m,2048,block,num_warps=4,num_stages=1)
  key=str(m)
  if key not in record['native_kernels']:
   binary=hashlib.sha256(compiled.kernel).hexdigest();assert binary==selected['expected_dequant_binaries'][key];record['native_kernels'][key]=dict(binary_sha256=binary,component_binary_exact_match=True,cache_hash=compiled.hash,n_regs=compiled.n_regs,n_spills=compiled.n_spills)
  return out
 def bf16(x,weight):
  name=next((name for name in ("_iluvatar.ops.mm","flag_gems.runtime.backend._iluvatar.ops.mm") if name in sys.modules),None);assert name is not None,"Qualified Iluvatar MM module not loaded"
  module=sys.modules[name];path=Path(module.__file__).resolve();digest=sha(path);assert digest=="4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090"
  if name not in record["BF16_vendor_modules"]:
   launch=Path(inspect.getsourcefile(module._launch_mm)).resolve();record["BF16_vendor_modules"][name]=dict(source=str(path),source_sha256=digest,launch_source=str(launch),launch_source_sha256=sha(launch),actual_vendor_mm_direct=True)
  return module.mm(x,weight.T)
 original_process=UnquantizedLinearMethod.process_weights_after_loading;original_apply=UnquantizedLinearMethod.apply
 @functools.wraps(original_process)
 def process(method,layer):
  nonlocal active,warmed
  result=original_process(method,layer)
  if tuple(layer.weight.shape)!=(2048,6144):return result
  prefix=layer.prefix;assert re.fullmatch(r'model\.layers\.\d+\.mlp\.down_proj',prefix) and prefix in expected
  weight=layer.weight;assert weight.is_cuda and weight.dtype==torch.bfloat16 and weight.is_contiguous() and not hasattr(layer,'_tianshu_native_down_bank');version=int(weight._version);proof=active_profiler();original_sha=thash(weight);assert original_sha==expected[prefix]['original_BF16_sha256']
  nt=torch.empty(weight.shape,device=weight.device,dtype=torch.int8);scales=torch.empty(2048,device=weight.device,dtype=torch.float32);packer.pack_into(weight,nt,scales,dense);packed=nt.T.contiguous();del nt;torch.cuda.synchronize();assert thash(packed.T.contiguous())==expected[prefix]['packed_sha256'] and thash(scales)==expected[prefix]['scales_sha256'] and int(weight._version)==version
  bankid=len(banks);layer.register_buffer('_tianshu_native_down_packed',packed,persistent=False);layer.register_buffer('_tianshu_native_down_scale',scales,persistent=False);layer._tianshu_native_down_bank=bankid;banks[bankid]=dict(method=method,layer=layer,prefix=prefix)
  payload=packed.numel()+scales.numel()*4;record['packed_payload_bytes']+=payload;record['banks'].append(dict(bank_id=bankid,prefix=prefix,original_BF16_sha256=original_sha,packed_sha256=expected[prefix]['packed_sha256'],scales_sha256=expected[prefix]['scales_sha256'],packed_shape=list(packed.shape),packed_stride=list(packed.stride()),payload_bytes=payload,inside_DeviceMemoryProfiler=proof,original_weight_version_preserved=True));save()
  if len(banks)==42:
   assert not warmed;cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();bank0=banks[0]['layer']
   for m in (64,128,512,4096):
    context=active_profiler();x=torch.zeros((m,6144),device=weight.device,dtype=torch.bfloat16);out=compute(x,bank0._tianshu_native_down_packed,bank0._tianshu_native_down_scale);torch.cuda.synchronize();assert bool((out==0).all());record['warmups'].append(dict(M=m,zero_output=True,inside_DeviceMemoryProfiler=context,INT32_workspace_bytes=m*2048*4,binary_sha256=record['native_kernels'][str(m)]['binary_sha256']));del x,out
   assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());warmed=active=True;record.update(phase='before_profile',warm_CPU_GPU_RNG_unchanged=True,load_complete=True);save()
  return result
 UnquantizedLinearMethod.process_weights_after_loading=process
 @torch.library.custom_op('tianshu_native_down::linear',mutates_args=(),device_types='cuda')
 def linear(x:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
  bank=banks[bank_id];m=x.shape[0];capturing=torch.cuda.is_current_stream_capturing();use=active and mode=='candidate' and m in record['supported_rows'] and x.is_contiguous();implementation='native_A8W8_Down' if use else 'original_BF16_Down';key=str((bank_id,m,capturing,record['phase'],implementation));row=record['routes'].setdefault(key,dict(prefix=bank['prefix'],bank_id=bank_id,M=m,K=6144,N=2048,phase=record['phase'],capturing=capturing,implementation=implementation,input_stride=list(x.stride()),calls=0));row['calls']+=1
  if use:
   assert x.shape==(m,6144) and x.dtype==torch.bfloat16 and packed.shape==(6144,2048) and packed.is_contiguous();out=compute(x,packed,scales);row.update(binary_sha256=record['native_kernels'][str(m)]['binary_sha256'],INT32_workspace_bytes=m*2048*4)
  else:out=bf16(x,weight)
  if bank_id==41 and row['calls']&(row['calls']-1)==0:save()
  return out
 @linear.register_fake
 def fake(x,weight,packed,scales,bank_id):return torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=x.dtype)
 @functools.wraps(original_apply)
 def apply(method,layer,x,bias=None):
  if bias is None and hasattr(layer,'_tianshu_native_down_bank') and x.ndim==2 and x.dtype==torch.bfloat16 and x.stride(-1)==1:return linear(x,layer.weight,layer._tianshu_native_down_packed,layer._tianshu_native_down_scale,int(layer._tianshu_native_down_bank))
  return original_apply(method,layer,x,bias)
 UnquantizedLinearMethod.apply=apply
 original_profile=ModelRunnerFL.profile_run
 @functools.wraps(original_profile)
 def profile(owner,*args,**kwargs):
  previous=record['phase'];record['phase']='profile';frame=inspect.currentframe().f_back;context=None
  try:
   while frame:
    if frame.f_code.co_name=='determine_available_memory' and frame.f_globals.get('__name__')=='vllm_fl.worker.worker':context=dict(source=str(Path(frame.f_code.co_filename).resolve()),sha256=sha(frame.f_code.co_filename),native_memory_context_active=('profile_result' in frame.f_locals),manual_KV_bytes=worker.cache_config.kv_cache_memory_bytes);break
    frame=frame.f_back
  finally:del frame
  item=dict(context=context,native_calls_before=sum(r['calls'] for r in record['routes'].values() if r['phase']=='profile' and r['implementation']=='native_A8W8_Down'),allocated_before=int(torch.cuda.memory_allocated()),peak_before=int(torch.cuda.max_memory_allocated()))
  try:return original_profile(owner,*args,**kwargs)
  finally:
   torch.cuda.synchronize();item.update(native_calls_after=sum(r['calls'] for r in record['routes'].values() if r['phase']=='profile' and r['implementation']=='native_A8W8_Down'),allocated_after=int(torch.cuda.memory_allocated()),peak_after=int(torch.cuda.max_memory_allocated()));record['profile_windows'].append(item);record['phase']=previous;save()
 ModelRunnerFL.profile_run=profile
 original_capture=ModelRunnerFL.capture_model
 @functools.wraps(original_capture)
 def capture(owner,*args,**kwargs):
  record['phase']='capture'
  try:result=original_capture(owner,*args,**kwargs)
  except BaseException:save();raise
  assert warmed and len(banks)==42;record.update(phase='runtime',capture_completed=True,completed=True,model_memory_usage_bytes=int(owner.model_memory_usage));save()
  if mode=='candidate':
   calls=[r for r in record['routes'].values() if r['phase']=='profile' and r['M']==4096 and r['implementation']=='native_A8W8_Down'];assert len({r['bank_id'] for r in calls})==42 and any(w['context'] and w['context']['native_memory_context_active'] and not w['context']['manual_KV_bytes'] and w['native_calls_after']-w['native_calls_before']>=42 for w in record['profile_windows'])
  return result
 ModelRunnerFL.capture_model=capture;atexit.register(save);save()
