# SPDX-License-Identifier: Apache-2.0
import atexit,functools,hashlib,importlib.util,inspect,json,os,re,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def install(worker,mode,directory):
 import torch
 from vllm.model_executor.layers.linear import UnquantizedLinearMethod
 from vllm_fl.worker.model_runner import ModelRunnerFL
 assert mode in ('control','candidate') and worker.vllm_config.speculative_config is None and worker.vllm_config.model_config.model=='/workspace/MiniCPM5-2B'
 assert worker.vllm_config.scheduler_config.max_num_batched_tokens==4096 and worker.vllm_config.scheduler_config.max_num_seqs==64 and worker.cache_config.kv_cache_memory_bytes is None
 selection=json.loads((BASE/'selected_plan.json').read_text());assert selection['complete'] and selection['quality_not_yet_established'] and selection['rows_by_role']=={'QKV':[128,512,4096],'O':[512,4096]};expected={b['prefix']:b for b in selection['banks']};assert len(expected)==84
 dense_path=EXP/'tianshu_dense_w8a8_20261020/kernel.py';epi_path=EXP/'tianshu_native_int8_lm64_20261022/kernel.py';pack_path=EXP/'tianshu_w8_gpu_packing_20261020/packing.py';dense=load('_native_attention_live_dense',dense_path);epi=load('_native_attention_live_epi',epi_path);packer=load('_native_attention_live_packer',pack_path);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);banks={};active=warmed=False
 record=dict(installed=True,mode=mode,phase='initializing',completed=False,capture_completed=False,banks=[],routes={},BF16_vendor_modules={},warmups=[],native_kernels={},profile_windows=[],packed_payload_bytes=0,one_NN_layout_per_bank=True,original_BF16_retained=True,no_global_tensor_workspace=True,Python_calls_are_not_Graph_replays=True,rows_by_role=selection['rows_by_role'],source=str(Path(__file__).resolve()),source_sha256=sha(__file__),dense_source=str(dense_path),dense_sha256=sha(dense_path),epilogue_source=str(epi_path),epilogue_sha256=sha(epi_path),packer_source=str(pack_path),packer_sha256=sha(pack_path),selection_source=str(BASE/'selected_plan.json'),selection_sha256=sha(BASE/'selected_plan.json'))
 def save():
  path=directory/f'route_{os.getpid()}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
 def thash(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 def profiler_context():
  frame=inspect.currentframe()
  try:
   while frame:
    if frame.f_globals.get('__name__')=='vllm_fl.worker.model_runner' and frame.f_code.co_name=='load_model' and 'm' in frame.f_locals:
     context=frame.f_locals['m'];assert hasattr(context,'initial_memory') and not hasattr(context,'final_memory');return dict(active=True,source=str(Path(frame.f_code.co_filename).resolve()),sha256=sha(frame.f_code.co_filename))
    frame=frame.f_back
   raise AssertionError('Attention packing/warmup outside DeviceMemoryProfiler')
  finally:del frame
 def compute(x,packed,scales,role):
  m,k=x.shape;n=packed.shape[1];assert m in selection['rows_by_role'][role] and k==2048;block=selection['block_by_M'][str(m)];q=torch.empty((m,k),device=x.device,dtype=torch.int8);asc=torch.empty(m,device=x.device,dtype=torch.float32);integer=torch.empty((m,n),device=x.device,dtype=torch.int32);out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16);dense.quantize(x,q,asc);torch._int_mm(q,packed,out=integer);compiled=epi._dequant[((m*n+block-1)//block,)](integer,asc,scales,out,m,n,block,num_warps=4,num_stages=1);key=f'{role}:{m}'
  if key not in record['native_kernels']:
   binary=hashlib.sha256(compiled.kernel).hexdigest();assert binary==selection['expected_dequant_binaries'][key];record['native_kernels'][key]=dict(binary_sha256=binary,component_binary_exact=True,cache_hash=compiled.hash,n_regs=compiled.n_regs,n_spills=compiled.n_spills)
  return out
 def bf16(x,weight):
  name=next((n for n in ('_iluvatar.ops.mm','flag_gems.runtime.backend._iluvatar.ops.mm') if n in sys.modules),None);assert name is not None;module=sys.modules[name];path=Path(module.__file__).resolve();assert sha(path)=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
  if name not in record['BF16_vendor_modules']:
   launch=Path(inspect.getsourcefile(module._launch_mm)).resolve();record['BF16_vendor_modules'][name]=dict(source=str(path),source_sha256=sha(path),launch_source=str(launch),launch_source_sha256=sha(launch),actual_vendor_mm_direct=True)
  return module.mm(x,weight.T)
 original_process=UnquantizedLinearMethod.process_weights_after_loading;original_apply=UnquantizedLinearMethod.apply
 @functools.wraps(original_process)
 def process(method,layer):
  nonlocal active,warmed
  result=original_process(method,layer);prefix=getattr(layer,'prefix','')
  if prefix not in expected:return result
  proof=expected[prefix];role=proof['role'];weight=layer.weight;assert tuple(weight.shape)==(proof['N'],2048) and weight.is_cuda and weight.dtype==torch.bfloat16 and weight.is_contiguous() and not hasattr(layer,'_tianshu_attention_bank');version=int(weight._version);context=profiler_context();assert thash(weight)==proof['original_BF16_sha256'];nt=torch.empty(weight.shape,device=weight.device,dtype=torch.int8);scales=torch.empty(proof['N'],device=weight.device,dtype=torch.float32);packer.pack_into(weight,nt,scales,dense);packed=nt.T.contiguous();del nt;torch.cuda.synchronize();assert thash(packed.T.contiguous())==proof['packed_sha256'] and thash(scales)==proof['scales_sha256'] and int(weight._version)==version
  bankid=len(banks);layer.register_buffer('_tianshu_attention_packed',packed,persistent=False);layer.register_buffer('_tianshu_attention_scales',scales,persistent=False);layer._tianshu_attention_bank=bankid;banks[bankid]=dict(layer=layer,prefix=prefix,role=role);payload=packed.numel()+scales.numel()*4;record['packed_payload_bytes']+=payload;record['banks'].append(dict(bank_id=bankid,prefix=prefix,role=role,N=proof['N'],original_BF16_sha256=proof['original_BF16_sha256'],packed_sha256=proof['packed_sha256'],scales_sha256=proof['scales_sha256'],packed_shape=list(packed.shape),packed_stride=list(packed.stride()),payload_bytes=payload,inside_DeviceMemoryProfiler=context,original_weight_version_preserved=True));save()
  if len(banks)==84:
   assert not warmed;cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
   for role,rows in selection['rows_by_role'].items():
    sample=next(bank['layer'] for bank in banks.values() if bank['role']==role)
    for m in rows:
     context=profiler_context();x=torch.zeros((m,2048),device=weight.device,dtype=torch.bfloat16);out=compute(x,sample._tianshu_attention_packed,sample._tianshu_attention_scales,role);torch.cuda.synchronize();assert bool((out==0).all());record['warmups'].append(dict(role=role,M=m,N=sample.weight.shape[0],zero_output=True,inside_DeviceMemoryProfiler=context,INT32_workspace_bytes=m*sample.weight.shape[0]*4,binary_sha256=record['native_kernels'][f'{role}:{m}']['binary_sha256']));del x,out
   assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());active=warmed=True;record.update(phase='before_profile',load_complete=True,warm_CPU_GPU_RNG_unchanged=True);save()
  return result
 UnquantizedLinearMethod.process_weights_after_loading=process
 @torch.library.custom_op('tianshu_native_attention::linear',mutates_args=(),device_types='cuda')
 def linear(x:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
  bank=banks[bank_id];role=bank['role'];m=x.shape[0];n=weight.shape[0];capturing=torch.cuda.is_current_stream_capturing();use=active and mode=='candidate' and m in selection['rows_by_role'][role] and x.is_contiguous();implementation='native_A8W8_attention_projection' if use else 'qualified_BF16_vendor_mm';key=str((bank_id,m,capturing,record['phase'],implementation));row=record['routes'].setdefault(key,dict(prefix=bank['prefix'],bank_id=bank_id,role=role,M=m,N=n,K=2048,phase=record['phase'],capturing=capturing,implementation=implementation,input_stride=list(x.stride()),calls=0));row['calls']+=1
  if use:out=compute(x,packed,scales,role);row.update(binary_sha256=record['native_kernels'][f'{role}:{m}']['binary_sha256'],INT32_workspace_bytes=m*n*4)
  else:out=bf16(x,weight)
  if bank_id==83 and row['calls']&(row['calls']-1)==0:save()
  return out
 @linear.register_fake
 def fake(x,weight,packed,scales,bank_id):return torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=x.dtype)
 @functools.wraps(original_apply)
 def apply(method,layer,x,bias=None):
  if bias is None and hasattr(layer,'_tianshu_attention_bank') and x.ndim==2 and x.dtype==torch.bfloat16 and x.stride(-1)==1:return linear(x,layer.weight,layer._tianshu_attention_packed,layer._tianshu_attention_scales,int(layer._tianshu_attention_bank))
  return original_apply(method,layer,x,bias)
 UnquantizedLinearMethod.apply=apply
 def native_profile_calls():return sum(r['calls'] for r in record['routes'].values() if r['phase']=='profile' and r['implementation']=='native_A8W8_attention_projection')
 original_profile=ModelRunnerFL.profile_run
 @functools.wraps(original_profile)
 def profile(owner,*args,**kwargs):
  previous=record['phase'];record['phase']='profile';frame=inspect.currentframe().f_back;context=None
  try:
   while frame:
    if frame.f_code.co_name=='determine_available_memory' and frame.f_globals.get('__name__')=='vllm_fl.worker.worker':context=dict(source=str(Path(frame.f_code.co_filename).resolve()),sha256=sha(frame.f_code.co_filename),native_memory_context_active=('profile_result' in frame.f_locals),manual_KV_bytes=worker.cache_config.kv_cache_memory_bytes);break
    frame=frame.f_back
  finally:del frame
  item=dict(context=context,native_calls_before=native_profile_calls(),allocated_before=int(torch.cuda.memory_allocated()),peak_before=int(torch.cuda.max_memory_allocated()))
  try:return original_profile(owner,*args,**kwargs)
  finally:torch.cuda.synchronize();item.update(native_calls_after=native_profile_calls(),allocated_after=int(torch.cuda.memory_allocated()),peak_after=int(torch.cuda.max_memory_allocated()));record['profile_windows'].append(item);record['phase']=previous;save()
 ModelRunnerFL.profile_run=profile
 original_capture=ModelRunnerFL.capture_model
 @functools.wraps(original_capture)
 def capture(owner,*args,**kwargs):
  record['phase']='capture'
  try:result=original_capture(owner,*args,**kwargs)
  except BaseException:save();raise
  assert warmed and len(banks)==84;record.update(phase='runtime',completed=True,capture_completed=True,model_memory_usage_bytes=int(owner.model_memory_usage));save()
  if mode=='candidate':
   calls=[r for r in record['routes'].values() if r['phase']=='profile' and r['M']==4096 and r['implementation']=='native_A8W8_attention_projection'];assert len({r['bank_id'] for r in calls})==84 and any(w['context'] and w['context']['native_memory_context_active'] and not w['context']['manual_KV_bytes'] and w['native_calls_after']-w['native_calls_before']>=84 for w in record['profile_windows'])
  return result
 ModelRunnerFL.capture_model=capture;atexit.register(save);save()
