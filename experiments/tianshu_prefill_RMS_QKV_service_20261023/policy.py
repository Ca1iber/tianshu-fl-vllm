# SPDX-License-Identifier: Apache-2.0
import atexit,functools,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def install(worker,directory):
 import torch
 from vllm_fl.worker.model_runner import ModelRunnerFL
 cfg=worker.vllm_config;assert cfg.model_config.model=='/workspace/MiniCPM5-2B' and cfg.speculative_config is None and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.cache_config.kv_cache_memory_bytes is None
 parent=load('_prefill_RMS_QKV_parent_bindings',BASE/'parent_bindings.py').bind_original_attention();old=parent['record'];banks=parent['banks'];old_save=parent['save'];component=EXP/'tianshu_prefill_RMS_A8_QKV_20261023';gate=json.loads((component/'INDEPENDENT_CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['component_lead'] and gate['numeric_cases']==18 and gate['full_K_FP64_norm_cells']==84934656 and gate['full_K_INT64_projection_reference_cells']==4608 and gate['fresh_model_C8_C64_quality_required']
 norm_path=EXP/'tianshu_IR_RMS_A8_20261023/kernel.py';epi_path=EXP/'tianshu_native_int8_lm64_20261022/kernel.py';norm=load('_prefill_RMS_QKV_fixed_numerical_kernel',norm_path);epi=load('_prefill_RMS_QKV_original_epilogue',epi_path)
 expected_norm={str(has):gate['compiled'][f'fused_norm_M4096_residual_{has}']['binary_sha256'] for has in (False,True)};expected_epi=gate['compiled']['QKV_dequant_M4096']['binary_sha256'];directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
 cc=cfg.compilation_config;old_sizes=list(cc.compile_sizes or []);assert old_sizes==[64];cc.compile_sizes=sorted(set(old_sizes)|{4096})
 record=dict(installed=True,mode='candidate',completed=False,routes={},profile_probes=[],original_compile_sizes=old_sizes,actual_compile_sizes=list(cc.compile_sizes),old_compile_size_record_scope='original parent hooks before this new hook; actual overall sizes recorded separately',source=str(BASE/'policy.py'),source_sha256=sha(__file__),norm_source=str(norm_path),norm_sha256=sha(norm_path),epilogue_source=str(epi_path),epilogue_sha256=sha(epi_path),parent_binding_source=str(BASE/'parent_bindings.py'),parent_binding_sha256=sha(BASE/'parent_bindings.py'),component_gate=str(component/'INDEPENDENT_CPU_GATE_32g.json'),component_gate_sha256=sha(component/'INDEPENDENT_CPU_GATE_32g.json'),expected_norm_binaries=expected_norm,expected_epilogue_binary=expected_epi,no_new_packed_weight_payload=True,no_global_tensor_workspace=True,QKV_other_M_O_Down_GU_LM_routes_unchanged=True,compatibility_index_only_actual_same_native_MM_and_epilogue_not_old_custom_op_or_A8=True,Python_calls_are_not_Graph_replays=True)
 def save():
  path=directory/f'route_{os.getpid()}.json';temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)
 def compute(x,residual,norm_weight,weight,packed,scales,bank_id,profile_probe=False):
  assert len(banks)==84 and old['load_complete'] and isinstance(bank_id,int) and bank_id in range(0,84,2);bank=banks[bank_id];layer=bank['layer'];has=residual is not None;assert has==(bank_id!=0) and bank['role']=='QKV'
  assert x.shape==(4096,2048) and x.dtype==torch.bfloat16 and x.is_contiguous() and norm_weight.shape==(2048,) and norm_weight.dtype==torch.bfloat16 and norm_weight.is_contiguous()
  assert not has or (residual.shape==x.shape and residual.dtype==x.dtype and residual.is_contiguous());assert weight.shape==(2560,2048) and weight.dtype==torch.bfloat16 and weight.data_ptr()==layer.weight.data_ptr() and packed.data_ptr()==layer._tianshu_attention_packed.data_ptr() and scales.data_ptr()==layer._tianshu_attention_scales.data_ptr() and packed.shape==(2048,2560) and packed.is_contiguous() and packed.dtype==torch.int8 and scales.shape==(2560,) and scales.dtype==torch.float32
  y=torch.empty_like(x);rr=torch.empty_like(residual) if has else None;q=torch.empty_like(x,dtype=torch.int8);ascale=torch.empty((4096,),device=x.device,dtype=torch.float32);integer=torch.empty((4096,2560),device=x.device,dtype=torch.int32);out=torch.empty((4096,2560),device=x.device,dtype=x.dtype)
  nc=norm.run(x,residual,norm_weight,y,rr,q,ascale);torch._int_mm(q,packed,out=integer);ec=epi._dequant[(4096*2560//512,)](integer,ascale,scales,out,4096,2560,512,num_warps=4,num_stages=1)
  capturing=torch.cuda.is_current_stream_capturing();phase='profile_probe' if profile_probe else old['phase'];trace=sys.modules.get('_tianshu_long_segments_actual_graph_trace');graph_key=getattr(trace,'STATE',{}).get('active_graph');key=str((bank_id,phase,capturing,graph_key));row=record['routes'].setdefault(key,dict(bank_id=bank_id,prefix=bank['prefix'],M=4096,K=2048,N=2560,has_residual=has,phase=phase,capturing=capturing,graph_key=graph_key,calls=0,explicit_zero_profile_probe=profile_probe,standalone_A8_quantizer_executed=False,norm_A8_fused=True,actual_original_weight_and_NN_pack_pointers_verified=True))
  if 'norm_binary_sha256' not in row:
   nh=hashlib.sha256(nc.kernel).hexdigest();eh=hashlib.sha256(ec.kernel).hexdigest();assert nh==expected_norm[str(has)] and eh==expected_epi;row.update(norm_binary_sha256=nh,epilogue_binary_sha256=eh,norm_cache_hash=nc.hash,epilogue_cache_hash=ec.hash,norm_component_binary_exact=True,epilogue_component_binary_exact=True)
  row['calls']+=1
  if not profile_probe:
   oldkey='actual_native_MM_from_prefill_RMS_QKV:'+key;previous=old['routes'].setdefault(oldkey,dict(prefix=bank['prefix'],bank_id=bank_id,role='QKV',M=4096,N=2560,K=2048,phase=phase,capturing=capturing,graph_key=graph_key,implementation='native_A8W8_attention_projection',input_stride=list(x.stride()),calls=0,binary_sha256=row['epilogue_binary_sha256'],INT32_workspace_bytes=4096*2560*4,wrapper_source=record['source'],wrapper_sha256=record['source_sha256'],standalone_A8_quantizer_executed=False,norm_A8_fused=True,scope='actual same native INT8 MM and dequant; not original custom_op or standalone A8 invocation'));previous['calls']+=1
  if bank_id==82 and row['calls']&(row['calls']-1)==0:save();old_save()
  return out,rr
 @torch.library.custom_op('tianshu_prefill_RMS_QKV::without_residual',mutates_args=(),device_types='cuda')
 def without_residual(x:torch.Tensor,norm_weight:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
  out,rr=compute(x,None,norm_weight,weight,packed,scales,bank_id);assert rr is None;return out
 @without_residual.register_fake
 def fake_without(x,norm_weight,weight,packed,scales,bank_id):return torch.empty((4096,2560),device=x.device,dtype=x.dtype)
 @torch.library.custom_op('tianshu_prefill_RMS_QKV::with_residual',mutates_args=(),device_types='cuda')
 def with_residual(x:torch.Tensor,residual:torch.Tensor,norm_weight:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->tuple[torch.Tensor,torch.Tensor]:
  out,rr=compute(x,residual,norm_weight,weight,packed,scales,bank_id);assert rr is not None;return out,rr
 @with_residual.register_fake
 def fake_with(x,residual,norm_weight,weight,packed,scales,bank_id):return torch.empty((4096,2560),device=x.device,dtype=x.dtype),torch.empty_like(residual)
 previous=cc.inductor_compile_config['post_grad_custom_post_pass'];fx=load('_prefill_RMS_QKV_static4096_pass',BASE/'fx_pass.py');cc.inductor_compile_config['post_grad_custom_post_pass']=fx.FusionPass(previous,directory/'fx_pass')
 def profile_context():
  frame=inspect.currentframe().f_back
  try:
   while frame:
    if frame.f_code.co_name=='determine_available_memory' and frame.f_globals.get('__name__')=='vllm_fl.worker.worker':return dict(source=str(Path(frame.f_code.co_filename).resolve()),sha256=sha(frame.f_code.co_filename),native_memory_context_active='profile_result' in frame.f_locals,manual_KV_bytes=worker.cache_config.kv_cache_memory_bytes)
    frame=frame.f_back
   raise AssertionError('New scratch probe outside native worker memory profile')
  finally:del frame
 original_profile=ModelRunnerFL.profile_run
 @functools.wraps(original_profile)
 def profile(owner,*args,**kwargs):
  result=original_profile(owner,*args,**kwargs);context=profile_context();assert context['native_memory_context_active'] and context['manual_KV_bytes'] is None;modules=dict(owner.model.named_modules());cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
  for bank_id in (0,2):
   layer=banks[bank_id]['layer'];name=banks[bank_id]['prefix'].removesuffix('.self_attn.qkv_proj')+'.input_layernorm';matches=[(n,m) for n,m in modules.items() if n==name or n.endswith('.'+name)];assert len(matches)==1;norm_name,norm_layer=matches[0];x=torch.zeros((4096,2048),device=layer.weight.device,dtype=torch.bfloat16);r=torch.zeros_like(x) if bank_id else None;versions=[int(t._version) for t in (layer.weight,layer._tianshu_attention_packed,layer._tianshu_attention_scales,norm_layer.weight)];before=int(torch.cuda.memory_allocated());peak_before=int(torch.cuda.max_memory_allocated());out,rr=compute(x,r,norm_layer.weight,layer.weight,layer._tianshu_attention_packed,layer._tianshu_attention_scales,bank_id,True);torch.cuda.synchronize();assert bool((out==0).all()) and (rr is None or bool((rr==0).all()));assert versions==[int(t._version) for t in (layer.weight,layer._tianshu_attention_packed,layer._tianshu_attention_scales,norm_layer.weight)];record['profile_probes'].append(dict(bank_id=bank_id,M=4096,K=2048,N=2560,has_residual=bool(bank_id),norm_module=norm_name,context=context,explicit_zero_probe_not_real_request=True,zero_output=True,integer_workspace_bytes=4096*2560*4,allocated_before=before,peak_before=peak_before,peak_after=int(torch.cuda.max_memory_allocated()),original_weights_versions_preserved=True));del x,r,out,rr
  assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state());record['profile_probe_CPU_GPU_RNG_unchanged']=True;save();old_save();return result
 ModelRunnerFL.profile_run=profile
 original_capture=ModelRunnerFL.capture_model
 @functools.wraps(original_capture)
 def capture(owner,*args,**kwargs):
  value=original_capture(owner,*args,**kwargs);real=[v for v in record['routes'].values() if not v['explicit_zero_profile_probe']];assert {v['bank_id'] for v in real}==set(range(0,84,2)) and all(v['norm_component_binary_exact'] and v['epilogue_component_binary_exact'] for v in real);assert record['profile_probes'] and record['profile_probe_CPU_GPU_RNG_unchanged'] and list(cfg.compilation_config.compile_sizes)==[64,4096];record.update(completed=True,all42_static4096_real_warm_invocations_verified=True,model_memory_usage_bytes=int(owner.model_memory_usage),actual_compile_sizes_at_capture=list(cfg.compilation_config.compile_sizes));save();old_save();return value
 ModelRunnerFL.capture_model=capture;atexit.register(save);save()
