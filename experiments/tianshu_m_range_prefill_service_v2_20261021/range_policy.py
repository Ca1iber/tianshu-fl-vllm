# SPDX-License-Identifier: Apache-2.0
"""Use qualified per-M prefills; retain original linear + generated activation elsewhere."""
import atexit,functools,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
KERNEL=EXP/'tianshu_nn_silu_large_tiles_v2_20261021/kernel.py'
ACTIVATION=EXP/'tianshu_m_range_mlp42_v2_20261021/inductor_activation.py'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def install(worker,mode,directory):
 import torch
 from vllm_fl.worker.model_runner import ModelRunnerFL
 from range_pass import RangePass
 assert mode in ('control','candidate');cfg=worker.vllm_config
 assert cfg.speculative_config is None and cfg.model_config.model=='/workspace/MiniCPM5-2B'
 assert cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64
 selection=json.loads((BASE/'selected_plan.json').read_text());component=Path(selection['component_gate']);gate=json.loads(component.read_text())
 assert sha(component)==selection['component_gate_sha256'] and gate['complete'] and gate['performance_valid']
 assert gate['all_3024_individual_graph_eager_checks_passed'] and gate['all_24_whole_chain_poison_checks_passed']
 assert all(sha(path)==digest for path,digest in gate['source_hashes'].items())
 configs={int(m):value for m,value in selection['configuration_by_M'].items()};assert 4096 in configs
 kernel=load('_range_live_fused_kernel',KERNEL);native=load('_range_live_quantizer',EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py');activation=load('_range_live_generated_activation',ACTIVATION)
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
 record=dict(mode=mode,installed=True,completed=False,active=False,no_additional_packed_weight_payload=True,precision_M_whitelist_not_expanded=True,M64_existing_fusion_preserved=True,compile_sizes=list(cfg.compilation_config.compile_sizes or []),configuration_by_M=configs,component_gate=str(component),component_gate_sha256=sha(component),selection_source=str(BASE/'selected_plan.json'),selection_sha256=sha(BASE/'selected_plan.json'),policy_source=str(BASE/'range_policy.py'),policy_sha256=sha(__file__),kernel_source=str(KERNEL),kernel_sha256=sha(KERNEL),activation_source=str(ACTIVATION),activation_sha256=sha(ACTIVATION),quantizer_source=str(EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py'),quantizer_sha256=sha(EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py'),routes={},warmups=[],actual_kernel_by_M={},activation_current_stream_each_call=True)
 active=False;dummy=None
 def save():
  path=directory/f'route_{os.getpid()}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
 def activate(proj,out,m):activation.triton_poi_fused_mul_silu_slice_1.run(proj,out,m*6144,stream=torch._C._cuda_getCurrentRawStream(0))
 def emit(row):
  row['calls']+=1
  if not row['capturing'] and row['prefix']=='model.layers.41.mlp.gate_up_proj' and row['calls']&(row['calls']-1)==0:save()
 # Unused Projected must retain 2D BF16 SME eligibility: reuse output, no storage.
 def compiled_record(m,compiled):
  if str(m) not in record['actual_kernel_by_M']:
   binary=hashlib.sha256(compiled.kernel).hexdigest();expected=gate['actual_timed_candidate_kernels'][str(m)]['binary_sha256'];assert binary==expected
   record['actual_kernel_by_M'][str(m)]=dict(binary_sha256=binary,expected_component_binary=expected,component_binary_exact_match=True,cache_hash=compiled.hash,registers=compiled.n_regs,spills=compiled.n_spills)
 @torch.library.custom_op('tianshu_m_range_prefill::fused_or_original',mutates_args=(),device_types='cuda')
 def fused_or_original(x:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
  assert x.ndim==2 and x.dtype==torch.bfloat16 and x.shape[1]==2048 and x.stride(-1)==1
  bank=sys.modules['_nn_silu_fx_NN_policy'].POINTERS[packed.data_ptr()];assert bank['bank_id']==bank_id
  m=x.shape[0];capturing=torch.cuda.is_current_stream_capturing();use=mode=='candidate' and active and m in configs and x.is_contiguous()
  implementation='M_range_two_dot_fused' if use else 'existing_M64_fused' if m==64 else 'original_linear_generated_activation'
  key=str((bank_id,m,active,capturing,implementation));row=record['routes'].setdefault(key,dict(prefix=bank['prefix'],bank_id=bank_id,M=m,capturing=capturing,stage='active' if active else 'before_activation',implementation=implementation,input_stride=list(x.stride()),calls=0))
  if m==64:
   out=torch.ops.tianshu_nn_silu_fx.fused.default(x,weight,packed,scales,bank_id);emit(row);return out
  if not use:
   proj=torch.ops.tianshu_prefill_int8.linear.default(x,weight,packed,scales,bank_id);out=torch.empty((m,6144),device=x.device,dtype=x.dtype);activate(proj,out,m);emit(row);return out
  assert packed.shape==(12288,2048) and packed.stride()==(1,12288) and packed.T.is_contiguous() and dummy is not None
  q=torch.empty((m,2048),device=x.device,dtype=torch.int8);s=torch.empty(m,device=x.device,dtype=torch.float32);out=torch.empty((m,6144),device=x.device,dtype=torch.bfloat16)
  native.quantize(x,q,s);compiled=kernel.fused(q,packed.T,s,scales,out,dummy,out,configs[m]);compiled_record(m,compiled)
  row.update(configuration=configs[m],kernel_source=str(KERNEL),kernel_sha256=record['kernel_sha256'],binary_sha256=record['actual_kernel_by_M'][str(m)]['binary_sha256']);emit(row);return out
 @fused_or_original.register_fake
 def fake(x,weight,packed,scales,bank_id):return torch.empty((x.shape[0],6144),device=x.device,dtype=x.dtype)
 cc=cfg.compilation_config;parent_pass=cc.inductor_compile_config['post_grad_custom_post_pass'];old_sizes=list(cc.compile_sizes or [])
 cc.inductor_compile_config['post_grad_custom_post_pass']=RangePass(parent_pass,mode,directory/'fx_pass',BASE/'selected_plan.json')
 assert list(cc.compile_sizes or [])==old_sizes;record['compile_sizes_unchanged']=True
 original_capture=ModelRunnerFL.capture_model
 @functools.wraps(original_capture)
 def capture(owner,*args,**kwargs):
  nonlocal active,dummy
  nn=sys.modules['_nn_silu_fx_NN_policy'];assert len(nn.LAYERS)==42
  layer=next(layer for layer in nn.LAYERS.values() if layer.prefix=='model.layers.0.mlp.gate_up_proj')
  before_cpu,before_gpu=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
  if mode=='candidate':
   p=torch.zeros((8,12288),device=layer.weight.device,dtype=torch.bfloat16);out=torch.empty((8,6144),device=p.device,dtype=p.dtype);activate(p,out,8);torch.cuda.synchronize();assert bool((out==0).all());del p,out
  assert torch.equal(torch.get_rng_state(),before_cpu) and torch.equal(torch.cuda.get_rng_state(),before_gpu);record['fallback_activation_warm_before_capture_RNG_unchanged']=True;record['fallback_activation_warm_performed']=(mode=='candidate')
  result=original_capture(owner,*args,**kwargs)
  before_cpu,before_gpu=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone();threads=torch.get_num_threads();started=time.perf_counter()
  if mode=='candidate':
   dummy=torch.empty(1,device=layer.weight.device,dtype=torch.int32);columns=torch.linspace(0,6143,16).long();full_columns=torch.cat((columns,columns+6144));cpu_w=layer._tianshu_dense_w8_packed.detach().cpu().contiguous()[full_columns];cpu_s=layer._tianshu_dense_w8_scale.detach().cpu()[full_columns]
   try:
    torch.set_num_threads(4)
    with torch.inference_mode():
     for m,config in sorted(configs.items()):
      cpu_a=((((torch.arange(m,dtype=torch.int32)[:,None]%17)-8)*((torch.arange(2048,dtype=torch.int32)[None,:]%13)-6))/8.).to(torch.bfloat16);cpu_a[0].zero_();a=cpu_a.to(layer.weight.device);q=torch.empty((m,2048),device=a.device,dtype=torch.int8);s=torch.empty(m,device=a.device,dtype=torch.float32);out=torch.empty((m,6144),device=a.device,dtype=torch.bfloat16)
      for _ in range(2):native.quantize(a,q,s);compiled=kernel.fused(q,layer._tianshu_dense_w8_packed.T,s,layer._tianshu_dense_w8_scale,out,dummy,out,config)
      torch.cuda.synchronize();compiled_record(m,compiled);v=cpu_a.float();asc=v.abs().amax(1).clamp_min(1.e-8)/127.;aq=(v/asc[:,None]).round().clamp(-127,127).to(torch.int8);assert torch.equal(q.cpu(),aq) and torch.equal(s.cpu(),asc)
      rows=torch.tensor(sorted({0,1,m//2,m-1}));dot=aq[rows].to(torch.int64)@cpu_w.to(torch.int64).T;proj=(dot.float()*asc[rows,None]*cpu_s[None,:]).to(torch.bfloat16);g,u=proj[:,:16].float(),proj[:,16:].float();expected=g/(1+torch.exp(-g))*u;actual=out[rows.to(a.device)[:,None],columns.to(a.device)[None,:]].cpu();assert torch.isfinite(actual).all() and ((actual.float()-expected).abs()<=.001+.02*expected.abs()).all()
      record['warmups'].append(dict(M=m,configuration=config,all_A_quantization_exact=True,CPU_selected_real_W_INT64_activation_passed=True,selected_elements=actual.numel(),component_binary_exact_match=True));del a,q,s,out,cpu_a,aq,asc,actual,expected,dot,proj
   finally:torch.set_num_threads(threads)
  assert torch.equal(torch.get_rng_state(),before_cpu) and torch.equal(torch.cuda.get_rng_state(),before_gpu)
  active=True;record.update(active=True,completed=True,warm_RNG_unchanged=True,CPU_threads_restored=torch.get_num_threads()==threads,warm_seconds=time.perf_counter()-started,activated_after_qualified_capture_and_warm=True)
  torch.cuda.empty_cache();save();return result
 ModelRunnerFL.capture_model=capture;atexit.register(save);save()
