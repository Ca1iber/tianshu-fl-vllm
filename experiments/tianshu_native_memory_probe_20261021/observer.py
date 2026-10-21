# SPDX-License-Identifier: Apache-2.0
"""Read native budget after the original method; no extra warmup/reset/allocation."""
import atexit,functools,hashlib,json,os
from pathlib import Path

def budget_snapshot(owner,value,allocated_before,allocated_after,estimate_enabled):
    row=dict(original_KV_memory_bytes=value,requested_memory=int(owner.requested_memory),
        weights_memory=int(owner.model_runner.model_memory_usage),
        peak_activation_memory=int(owner.peak_activation_memory),
        non_torch_memory=int(owner.non_torch_memory),
        cudagraph_memory_estimate=int(owner.cudagraph_memory_estimate),
        cudagraph_estimate_enabled=bool(estimate_enabled),
        torch_allocated_before_native=allocated_before,torch_allocated_after_native=allocated_after,
        native_return_preserved=True,no_extra_profile_run=True,no_peak_reset=True,no_config_or_allocator_change=True)
    expected=row['requested_memory']-row['weights_memory']-row['peak_activation_memory']-row['non_torch_memory']
    if estimate_enabled:expected-=row['cudagraph_memory_estimate']
    assert expected==value
    row.update(native_budget_equation_exact=True,nominal_K8V16_42layer_blocks=value//516096)
    return row

def install(worker,mode,directory):
    import torch
    import vllm.envs as envs
    cfg=worker.vllm_config
    assert cfg.model_config.model=='/workspace/MiniCPM5-2B' and cfg.speculative_config is None and cfg.lora_config is None
    assert cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64
    assert sorted(cfg.compilation_config.compile_sizes)==([] if mode=='control' else [8,64])
    cls=type(worker);assert not getattr(cls,'_native_memory_observer_installed',False)
    original=cls.determine_available_memory;directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    path=Path(__file__).resolve()
    record=dict(mode=mode,complete=False,observer_source=str(path),observer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        precision_and_native_memory_method_unchanged=True,only_read_existing_native_fields=True)
    def save():
        p=directory/f'observed_{os.getpid()}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(p)
    @functools.wraps(original)
    def available(owner,*args,**kwargs):
        before=int(torch.cuda.memory_allocated());value=original(owner,*args,**kwargs)
        record['budget']=budget_snapshot(owner,value,before,int(torch.cuda.memory_allocated()),envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS)
        record['complete']=True;save();return value
    cls.determine_available_memory=available;cls._native_memory_observer_installed=True;atexit.register(save);save()
