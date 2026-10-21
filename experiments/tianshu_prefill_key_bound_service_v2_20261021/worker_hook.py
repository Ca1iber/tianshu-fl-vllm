# SPDX-License-Identifier: Apache-2.0
"""Install only after qualified pipeline, persist metadata at existing lifecycle points."""
import atexit
import functools
import hashlib
import inspect
import importlib.util
import json
import os
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
HYBRID_SHA='5c55a8d57e438b36d176555fd9273ab9ba665dada4c57d57c19633394c2ed4e4'
NATIVE_METADATA_SHA='c524988a2223ec556bdb128a31c255075f0d07e7643e803b447975cd9c30a6b8'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_dense(dense):
    cls=type(dense);parent=cls.__mro__[1]
    native=BASE.parent/'tianshu_k8v16_pipeline_20261019/bounded_prefill.py'
    wrapper=BASE.parent/'tianshu_small_arena_20261020/policy.py'
    assert cls.__name__=='CapacityBoundedDensePrefill' and parent.__name__=='BoundedDensePrefill'
    assert getattr(cls,'_small_arena_policy',False) and dense.limit_pages==8192
    assert Path(inspect.getsourcefile(cls)).resolve()==wrapper.resolve()
    assert Path(inspect.getsourcefile(parent)).resolve()==native.resolve()
    assert cls.__call__ is parent.__call__ and cls.verify_last is parent.verify_last
    assert Path(inspect.getsourcefile(cls.__call__)).resolve()==native.resolve()
    return dict(wrapper_class=cls.__name__,native_class=parent.__name__,capacity_pages=8192,
        actual_small_arena_wrapper_source=str(wrapper),actual_small_arena_wrapper_sha256=sha(wrapper),
        original_native_call_source=str(native),original_native_call_sha256=sha(native),
        original_call_and_verify_inherited=True,subclass_reserve_layout_CPU_verified=True)

def install(worker,mode,directory):
    import torch
    from vllm.platforms import current_platform
    from vllm_fl.worker import iluvatar_hybrid_attention as hybrid
    from vllm_fl.worker.model_runner import ModelRunnerFL
    import vllm.v1.attention.backends.triton_attn as backend
    import vllm.v1.attention.backend as metadata
    import vllm.envs as envs
    cfg=worker.vllm_config;hf=cfg.model_config.hf_config
    assert mode in ('control','candidate') and current_platform.vendor_name=='iluvatar'
    assert cfg.model_config.model=='/workspace/MiniCPM5-2B' and cfg.model_config.dtype==torch.bfloat16 and cfg.model_config.quantization is None
    assert cfg.speculative_config is None and cfg.lora_config is None
    assert (hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads)==(2048,42,16,2)
    assert cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==cfg.parallel_config.data_parallel_size==cfg.parallel_config.decode_context_parallel_size==1
    assert cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.async_scheduling
    assert cfg.cache_config.cache_dtype=='auto' and cfg.cache_config.block_size in (None,16) and cfg.cache_config.kv_cache_memory_bytes is None
    assert sha(hybrid.__file__)==HYBRID_SHA and sha(metadata.__file__)==NATIVE_METADATA_SHA and hybrid._INSTALLED
    pipeline=backend._soft8_pipeline_installed;dense=pipeline['dense']
    guard=check_dense(dense)
    policy_path=BASE.parent/'tianshu_prefill_key_bound_policy_20261021/policy.py'
    spec=importlib.util.spec_from_file_location('_actual_prefill_key_bound_stage_policy',policy_path)
    policy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=policy;spec.loader.exec_module(policy)
    state=policy.install(hybrid,dense,mode);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    record=dict(mode=mode,installed=True,actual_dense_guard=guard,capture_completed=False,actual_arena_pages=dense.limit_pages,
        worker_init_complete_before_install=True,only_prefill_bound_changed=True,no_parameter_or_KV_writer_or_sampling_changes=True,
        metadata_contract_source=str(Path(metadata.__file__).resolve()),metadata_contract_sha256=NATIVE_METADATA_SHA,
        native_route_source=str(Path(hybrid.__file__).resolve()),native_route_sha256=HYBRID_SHA,
        policy_source=str(policy_path),policy_sha256=sha(policy_path),hook_source=str(Path(__file__).resolve()),hook_sha256=sha(__file__),
        component_gate_source=str(BASE.parent/'tianshu_prefill_key_bound_results_20261021/COMPONENT_GATE_32g.json'),
        component_gate_sha256=sha(BASE.parent/'tianshu_prefill_key_bound_results_20261021/COMPONENT_GATE_32g.json'),
        counts=state,additional_per_step_file_writes=False,existing_engine_records_only_for_actual_stage_evidence=True)
    def save():
        path=directory/f'binding_{os.getpid()}.json';temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)
    original_available=type(worker).determine_available_memory
    @functools.wraps(original_available)
    def available(owner,*args,**kwargs):
        value=original_available(owner,*args,**kwargs)
        record['native_memory_budget']=dict(original_return_KV_memory_bytes=value,
            original_return_and_config_preserved=True,additional_profile_run_not_performed=True,
            peak_stats_not_reset_by_observer=True,
            weights_memory=int(owner.model_runner.model_memory_usage),
            native_peak_activation_memory=int(owner.peak_activation_memory),
            native_non_torch_memory=int(owner.non_torch_memory),
            native_cudagraph_memory_estimate=int(owner.cudagraph_memory_estimate),
            native_cudagraph_estimate_enabled=bool(envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS),
            current_torch_memory_allocated=int(torch.cuda.memory_allocated()))
        save();return value
    type(worker).determine_available_memory=available
    original_capture=ModelRunnerFL.capture_model
    @functools.wraps(original_capture)
    def capture(owner,*args,**kwargs):
        answer=original_capture(owner,*args,**kwargs);assert not record['capture_completed']
        state['stage']='active';record['capture_completed']=True;save();return answer
    ModelRunnerFL.capture_model=capture;atexit.register(save);save()
