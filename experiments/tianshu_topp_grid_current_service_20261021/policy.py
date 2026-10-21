# SPDX-License-Identifier: Apache-2.0
"""Equal pre-yield warmup; geometry becomes active after qualified parent capture."""
import atexit,functools,gc,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
STATE=dict(stage='before_install',ready=False,capture_returned=False)
spec=importlib.util.spec_from_file_location('_grid_profile_context',BASE/'profile_context.py')
context=importlib.util.module_from_spec(spec);spec.loader.exec_module(context)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def install(worker,mode,directory):
 assert mode in ('control','candidate')
 import torch
 import vllm_fl.worker.worker as worker_module
 from vllm_fl.worker.model_runner import ModelRunnerFL
 from vllm.v1.sample.ops import topk_topp_triton as native
 cfg=worker.vllm_config;hf=cfg.model_config.hf_config
 assert cfg.model_config.model=='/workspace/MiniCPM5-2B' and cfg.model_config.dtype==torch.bfloat16
 assert (hf.hidden_size,hf.num_hidden_layers,hf.vocab_size)==(2048,42,130560)
 assert cfg.speculative_config is None and cfg.lora_config is None
 assert cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.async_scheduling
 assert cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==cfg.parallel_config.data_parallel_size==1
 assert not getattr(ModelRunnerFL,'_tianshu_current_grid_installed',False)
 assert os.environ['TIANSHU_TOPP_PRECISION_SERVICE_MODE']=='candidate'
 selection=json.loads((BASE/'selected_plan.json').read_text());gate=json.loads(Path(selection['API_gate']).read_text())
 assert sha(selection['API_gate'])==selection['API_gate_sha256'] and gate['complete'] and gate['all_per_row_original_contract_passed']
 route=native._tianshu_grid_route;assert not route.enabled
 geo_path=Path(sys.modules['_tianshu_topp_geometric_geo'].__file__)
 assert sha(geo_path)=='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
 original_profile=worker_module.memory_profiling_fl;original_capture=ModelRunnerFL.capture_model
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 (directory/'actual_generated_geo.py').write_bytes(geo_path.read_bytes())
 record=dict(mode=mode,installed=True,completed=False,warm_complete=False,native_context_returned=False,capture_calls=0,warm_calls=[],
   policy_source=str(BASE/'policy.py'),policy_sha256=sha(__file__),context_source=str(BASE/'profile_context.py'),context_sha256=sha(BASE/'profile_context.py'),
   kernel_route_source=str(BASE/'kernel_route.py'),kernel_route_sha256=sha(BASE/'kernel_route.py'),selection_sha256=sha(BASE/'selected_plan.json'),
   actual_GEO_source=str(geo_path),actual_GEO_sha256=sha(geo_path),original_memory_profile_source=inspect.getsourcefile(inspect.unwrap(original_profile)),original_capture_source=inspect.getsourcefile(original_capture),
   actual_budget=4096,actual_max_seqs=64,parent_model_precision_and_Graphs_unchanged=True,original_Sampler_and_RNG_unchanged=True,
   no_target_model_Graph_changes=True,Python_counts_not_Graph_replay_counts=True)
 for key in ('original_memory_profile','original_capture'):record[key+'_sha256']=sha(record[key+'_source'])
 def save():
  target=directory/f'policy_{os.getpid()}.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(target)
 def tensor_sha(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
 def warm(result):
  assert not record['warm_complete'] and not STATE['capture_returned'] and not torch.cuda.is_current_stream_capturing()
  stack=[frame.function for frame in inspect.stack()];assert 'determine_available_memory' in stack
  STATE['stage']='profile_manual_warm'
  record.update(native_before_profile_peak=int(result.before_profile.torch_peak),native_weights_memory=int(result.weights_memory),profile_warm_stack=stack[:12],native_profile_entered_before_warm=True,warm_before_native_context_yield=True,no_manual_peak_reset=True)
  cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
  values=((torch.arange(130560,dtype=torch.float32)%1024)-512.)/128.
  source=values.repeat(64,1).cuda();work=torch.empty_like(source);p=torch.full((64,),.95,device='cuda',dtype=torch.float32)
  for enabled in (False,True):
   route.enabled=enabled
   for batch in range(8,65):
    work[:batch].copy_(source[:batch]);native.apply_top_k_top_p_triton(work[:batch],None,p[:batch])
    record['warm_calls'].append(dict(batch=batch,candidate_geometry=enabled,input_stride=[130560,1]))
  route.enabled=False;torch.cuda.synchronize()
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
  record.update(warm_CPU_RNG_before=tensor_sha(cpu_rng),warm_CPU_RNG_after=tensor_sha(torch.get_rng_state()),warm_GPU_RNG_before=tensor_sha(gpu_rng),warm_GPU_RNG_after=tensor_sha(torch.cuda.get_rng_state()),warm_default_RNG_unchanged=True,warm_both_arms_same=True,native_peak_after_warm_bytes=torch.cuda.max_memory_allocated())
  del source,work,p,values;gc.collect();torch.cuda.synchronize()
  inner=inspect.getclosurevars(route.__getitem__).nonlocals['inner']
  scratch=inner.__class__.__getitem__.__globals__['SCRATCH'];assert len(scratch)==1
  buffer=next(iter(scratch.values()));assert buffer.shape==(64,130560) and buffer.dtype==torch.float32 and buffer.is_cuda
  cache=native._TRITON_BUFFER_CACHE
  entries=[value for key,value in cache.items() if key[0]==torch.device('cuda',torch.cuda.current_device()) and key[1]==torch.float32 and key[2]==130560]
  assert len(entries)==1
  record.update(shared_sampler_scratch_shape=list(buffer.shape),shared_sampler_scratch_bytes=buffer.numel()*buffer.element_size(),native_sampler_buffer_shape=list(entries[0].shape),native_sampler_buffer_bytes=entries[0].numel()*entries[0].element_size(),warm_temporary_tensors_released=True,memory_allocated_after_warm=torch.cuda.memory_allocated(),warm_complete=True)
  STATE['stage']='pre_capture_native';native._tianshu_grid_flush();save()
 def finished(result):
  record.update(native_context_returned=True,native_context_peak_increase=int(result.torch_peak_increase),native_context_non_torch_increase=int(result.non_torch_increase));save()
 worker_module.memory_profiling_fl=context.with_warmup(original_profile,warm,finished)
 @functools.wraps(original_capture)
 def capture(owner,*args,**kwargs):
  answer=original_capture(owner,*args,**kwargs);record['capture_calls']+=1
  assert record['capture_calls']==1 and record['warm_complete'] and record['native_context_returned']
  assert not torch.cuda.is_current_stream_capturing()
  assert worker.peak_activation_memory+record['native_before_profile_peak']>=record['native_peak_after_warm_bytes']
  route.enabled=mode=='candidate';STATE.update(capture_returned=True,stage='post_capture_warm')
  record.update(completed=True,native_final_peak_activation_memory=int(worker.peak_activation_memory),native_final_non_torch_memory=int(worker.non_torch_memory),native_graph_estimate_bytes=int(worker.cudagraph_memory_estimate),native_activation_peak_covers_warm_peak=True,after_complete_parent_capture_returned=True,original_capture_return_preserved=True,runtime_candidate_geometry=(mode=='candidate'))
  save();return answer
 ModelRunnerFL.capture_model=capture;ModelRunnerFL._tianshu_current_grid_installed=True
 atexit.register(native._tianshu_grid_flush);atexit.register(save);save()
