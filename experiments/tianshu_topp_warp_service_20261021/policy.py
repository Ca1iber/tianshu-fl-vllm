# SPDX-License-Identifier: Apache-2.0
"""Prewarm both TopP warp variants inside the real native memory profile."""
import atexit,functools,gc,hashlib,inspect,json,os,importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("_topp_warp_route_contract",BASE/"route.py")
route=importlib.util.module_from_spec(spec);spec.loader.exec_module(route)
KernelRouter,profile_with_warmup=route.KernelRouter,route.profile_with_warmup

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def install(worker,mode,directory):
    assert mode in ('control','candidate')
    import torch
    import vllm_fl.worker.worker as worker_module
    from vllm_fl.worker.model_runner import ModelRunnerFL
    import vllm.v1.sample.ops.topk_topp_triton as native
    cfg=worker.vllm_config;hf=cfg.model_config.hf_config
    assert cfg.speculative_config is None and cfg.lora_config is None
    assert cfg.model_config.model=='/workspace/MiniCPM5-2B' and cfg.model_config.dtype==torch.bfloat16
    assert (hf.hidden_size,hf.num_hidden_layers,hf.vocab_size)==(2048,42,130560)
    assert cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.async_scheduling is True
    assert os.environ['TIANSHU_LM_PRECISION_MODE']=='candidate'
    assert not getattr(ModelRunnerFL,'_tianshu_topp_warp_installed',False)
    selection=json.loads((BASE/'selected_plan.json').read_text());gate=json.loads(Path(selection['component_gate']).read_text())
    assert sha(selection['component_gate'])==selection['component_gate_sha256'] and gate['complete'] and gate['CPU_reverified']
    assert selection['exploratory_non_bitwise_candidate'] and not gate['bitwise_native_all_cases_by_config']['warp16']
    original_kernel=native._topk_topp_kernel;kernel_source=Path(original_kernel.fn.__code__.co_filename).resolve()
    assert sha(kernel_source)==gate['frozen_sources'][str(kernel_source)]
    original_profile=worker_module.memory_profiling_fl;original_capture=ModelRunnerFL.capture_model
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    record=dict(mode=mode,installed=True,completed=False,profile_warm_complete=False,native_context_returned=False,capture_calls=0,active=False,request_ready_seen=False,warm_calls=[],routes={},policy_source=str(Path(__file__).resolve()),policy_sha256=sha(__file__),route_source=str(BASE/'route.py'),route_sha256=sha(BASE/'route.py'),kernel_source=str(kernel_source),kernel_sha256=sha(kernel_source),selection_sha256=sha(BASE/'selected_plan.json'),original_memory_profile_source=inspect.getsourcefile(inspect.unwrap(original_profile)),original_capture_source=inspect.getsourcefile(original_capture),actual_budget=4096,actual_max_seqs=64,default_warps=4,candidate_warps=16,FP32_reduction_differences_expected=True,default_RNG_algorithm_unchanged=True,no_target_model_Graph_changes=True,Python_counts_not_Graph_replay_counts=True)
    for field in ('original_memory_profile','original_capture'):record[field+'_sha256']=sha(record[field+'_source'])
    state=dict(force=None,active=False,ready=False,calls=0)
    def save():
        target=directory/f'route_{os.getpid()}.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(target)
    def selection_warps():return state['force'] if state['force'] is not None else (16 if mode=='candidate' and state['active'] else None)
    def observe(grid,args,kw,compiled):
        state['calls']+=1
        if state['force'] is not None:stage='profile_manual_warm'
        else:
            if state['active'] and not state['ready']:state['ready']=(directory.parent/'observed_server_environment.json').exists()
            stage='runtime' if state['active'] and state['ready'] else ('post_capture_warm' if state['active'] else 'pre_capture_native')
        logits=args[0];warps=kw.get('num_warps',4);key=f'{stage}_B{logits.shape[0]}_W{warps}'
        if key not in record['routes']:
            m=compiled.metadata
            record['routes'][key]=dict(stage=stage,batch=logits.shape[0],shape=list(logits.shape),stride=list(logits.stride()),dtype=str(logits.dtype),device=str(logits.device),num_warps=getattr(m,'num_warps',None),num_stages=getattr(m,'num_stages',None),n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),kernel_hash=compiled.hash,grid=list(grid),TOPK_ENABLED=kw['TOPK_ENABLED'],TOPP_ENABLED=kw['TOPP_ENABLED'],BLOCK_SIZE=kw['BLOCK_SIZE'],BLOCK_SIZE_TRUNC=kw['BLOCK_SIZE_TRUNC'],actually_inside_capture=torch.cuda.is_current_stream_capturing(),calls=0)
            assert record['routes'][key]['num_warps']==warps
            save()
        record['routes'][key]['calls']+=1;record['request_ready_seen']=state['ready']
        if state['calls']%256==0:save()
    native._topk_topp_kernel=KernelRouter(original_kernel,selection_warps,observe)
    def tensor_sha(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
    def warm(result):
        assert not record['profile_warm_complete'] and not state['active'] and not torch.cuda.is_current_stream_capturing()
        stack=[f.function for f in inspect.stack()];assert 'determine_available_memory' in stack
        record.update(profile_warm_scope='Before native context yield; original native context already entered; no manual peak reset.',native_before_profile_peak=int(result.before_profile.torch_peak),native_weights_memory=int(result.weights_memory),memory_allocated_before_warm=torch.cuda.memory_allocated(),profile_warm_stack=stack[:12]);save()
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        values=((torch.arange(130560,dtype=torch.float32)%1024)-512.)/128.
        source=values.repeat(64,1).cuda();work=torch.empty_like(source);p=torch.full((64,),.95,dtype=torch.float32,device='cuda')
        for warps in (4,16):
            state['force']=warps
            for batch in range(8,65):
                work[:batch].copy_(source[:batch]);native.apply_top_k_top_p_triton(work[:batch],None,p[:batch])
                record['warm_calls'].append(dict(batch=batch,num_warps=warps,input_stride=[130560,1]))
        state['force']=None;torch.cuda.synchronize()
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        record.update(warm_CPU_RNG_before=tensor_sha(cpu_rng),warm_CPU_RNG_after=tensor_sha(torch.get_rng_state()),warm_GPU_RNG_before=tensor_sha(gpu_rng),warm_GPU_RNG_after=tensor_sha(torch.cuda.get_rng_state()),warm_default_RNG_unchanged=True,native_peak_after_warm_bytes=torch.cuda.max_memory_allocated(),warm_both_arms_same=True)
        del source,work,p,values;gc.collect();torch.cuda.synchronize()
        cache=native._TRITON_BUFFER_CACHE;entries=[v for k,v in cache.items() if k[0]==torch.device('cuda',torch.cuda.current_device()) and k[1]==torch.float32 and k[2]==130560]
        assert len(entries)==1;record['native_sampler_buffer_shape']=list(entries[0].shape);record['native_sampler_buffer_bytes']=entries[0].numel()*entries[0].element_size()
        record.update(memory_allocated_after_warm=torch.cuda.memory_allocated(),warm_temporary_tensors_released=True,profile_warm_complete=True);save()
    def finished(result):
        record.update(native_context_returned=True,native_context_profile_torch_peak_increase=int(result.torch_peak_increase),native_context_non_torch_increase=int(result.non_torch_increase));save()
    worker_module.memory_profiling_fl=profile_with_warmup(original_profile,warm,finished)
    @functools.wraps(original_capture)
    def capture(owner,*args,**kwargs):
        answer=original_capture(owner,*args,**kwargs);record['capture_calls']+=1
        assert record['capture_calls']==1 and record['profile_warm_complete'] and record['native_context_returned']
        assert not torch.cuda.is_current_stream_capturing()
        assert worker.peak_activation_memory+record['native_before_profile_peak']>=record['native_peak_after_warm_bytes']
        record.update(native_final_peak_activation_memory=int(worker.peak_activation_memory),native_final_non_torch_memory=int(worker.non_torch_memory),native_graph_estimate_bytes=int(worker.cudagraph_memory_estimate),native_activation_peak_covers_warm_peak=True,after_complete_parent_capture_returned=True,original_capture_return_preserved=True,completed=True,active=True)
        state['active']=True;save();return answer
    ModelRunnerFL.capture_model=capture;ModelRunnerFL._tianshu_topp_warp_installed=True
    atexit.register(save);save()
