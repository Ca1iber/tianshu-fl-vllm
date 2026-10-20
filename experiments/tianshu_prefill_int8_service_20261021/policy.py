# SPDX-License-Identifier: Apache-2.0
"""Prefill A8/W8 after qualified capture, same NN decode in both arms."""
import atexit,functools,hashlib,importlib.util,json,os,re,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
KERNEL=BASE.parent/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install(worker,mode,directory):
    assert mode in ('control','candidate')
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm_fl.worker.model_runner import ModelRunnerFL
    cfg=worker.vllm_config
    assert cfg.model_config.model=='/workspace/MiniCPM5-2B' and cfg.model_config.quantization is None
    assert cfg.speculative_config is None and cfg.lora_config is None
    assert cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.async_scheduling
    assert cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==cfg.parallel_config.data_parallel_size==1
    assert os.environ['TIANSHU_LM_PRECISION_MODE']=='candidate' and os.environ['VLLM_DISABLE_COMPILE_CACHE']=='1'
    selection=json.loads((BASE/'selected_plan.json').read_text());assert selection['complete'] and selection['component_only'] and selection['kernel_sha256']==sha(KERNEL)
    component=Path(selection['component_gate']);gate=json.loads(component.read_text());assert sha(component)==selection['component_gate_sha256'] and gate['complete'] and gate['independently_CPU_rechecked']
    assert all(sha(p)==v for p,v in gate['frozen_sources'].items())
    selected={r['M']:r['configuration'] for r in selection['configurations']};assert set(selected)=={128,256,512,1024,1152,2048,4096}
    spec=importlib.util.spec_from_file_location('_prefill_int8_grouped_kernel',KERNEL);kernel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernel;spec.loader.exec_module(kernel)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    record=dict(mode=mode,installed=True,completed=False,stage='before_capture',banks=[],routes={},warmups=[],warm_configs={},
        policy_source=str(BASE/'policy.py'),policy_sha256=sha(__file__),kernel_source=str(KERNEL),kernel_sha256=sha(KERNEL),selection_sha256=sha(BASE/'selected_plan.json'),
        model_precision_changes_prefill=(mode=='candidate'),same_NN_decode_in_both_arms=True,LM_not_intercepted=True,no_new_packed_weight_payload=True,
        Python_launch_counts_are_not_Graph_replays=True,unknown_M_or_layout_falls_back_without_new_INT8_JIT=True,CPU_only_reference_scope='All warm A quantization and selected actual GPU BF16 outputs from real checkpoint bank; not full warm output CPU recomputation')
    banks={};warm_configs={};active=False;dummy=None
    original_process=UnquantizedLinearMethod.process_weights_after_loading;original_apply=UnquantizedLinearMethod.apply;original_capture=ModelRunnerFL.capture_model
    def save():
        path=directory/f'route_{os.getpid()}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
    def emit(row):
        row['calls']+=1
        if row['prefix']=='model.layers.41.mlp.gate_up_proj' and row['calls']&(row['calls']-1)==0:save()
    @functools.wraps(original_process)
    def process(method,layer):
        value=original_process(method,layer)
        if hasattr(layer,'_tianshu_dense_w8_bank') and tuple(layer.weight.shape)==(12288,2048):
            assert re.fullmatch(r'model\.layers\.\d+\.mlp\.gate_up_proj',layer.prefix)
            packed=layer._tianshu_dense_w8_packed;assert tuple(packed.stride())==(1,12288) and packed.T.is_contiguous()
            bank_id=int(layer._tianshu_dense_w8_bank);assert bank_id not in banks
            banks[bank_id]=dict(layer=layer,method=method,prefix=layer.prefix)
            record['banks'].append(dict(bank_id=bank_id,prefix=layer.prefix,logical_shape=list(packed.shape),logical_stride=list(packed.stride()),payload_bytes=packed.numel()*packed.element_size()))
            save()
        return value
    UnquantizedLinearMethod.process_weights_after_loading=process
    @torch.library.custom_op('tianshu_prefill_int8::linear',mutates_args=(),device_types='cuda')
    def linear(x:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
        bank=banks[bank_id];m=x.shape[0];capturing=torch.cuda.is_current_stream_capturing()
        use=active and mode=='candidate' and m in warm_configs and x.is_contiguous()
        implementation='prefill_A8W8' if use else 'delegate_qualified'
        key=str((bank_id,m,record['stage'],capturing,implementation))
        row=record['routes'].setdefault(key,dict(prefix=bank['prefix'],M=m,stage=record['stage'],actually_inside_capture=capturing,implementation=implementation,input_stride=list(x.stride()),calls=0))
        if not use:
            output=original_apply(bank['method'],bank['layer'],x,None);emit(row);return output
        assert 128<=m<=4096 and packed.stride()==(1,12288) and x.dtype==torch.bfloat16 and x.shape[1]==2048
        config=warm_configs[m];q=torch.empty((m,2048),device=x.device,dtype=torch.int8);s=torch.empty(m,device=x.device,dtype=torch.float32)
        out=torch.empty((m,12288),device=x.device,dtype=torch.bfloat16)
        kernel.quantize(x,q,s);compiled=kernel.mm(q,packed.T,s,scales,out,dummy,config)
        if 'kernel_sha256' not in row:
            row.update(configuration=config,kernel_source=str(KERNEL),kernel_sha256=sha(KERNEL),cache_hash=compiled.hash,binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared)
        emit(row);return out
    @linear.register_fake
    def fake(x,weight,packed,scales,bank_id):return torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=x.dtype)
    @functools.wraps(original_apply)
    def apply(method,layer,x,bias=None):
        if bias is not None or not hasattr(layer,'_tianshu_dense_w8_bank') or tuple(layer.weight.shape)!=(12288,2048) or x.ndim!=2 or x.dtype!=torch.bfloat16 or x.stride(-1)!=1:return original_apply(method,layer,x,bias)
        return linear(x,layer.weight,layer._tianshu_dense_w8_packed,layer._tianshu_dense_w8_scale,int(layer._tianshu_dense_w8_bank))
    UnquantizedLinearMethod.apply=apply
    @functools.wraps(original_capture)
    def capture(owner,*args,**kwargs):
        nonlocal active,dummy
        assert not active and not record['completed'];value=original_capture(owner,*args,**kwargs);assert len(banks)==42
        record.update(stage='warming',after_original_capture=True);save()
        previous_threads=torch.get_num_threads();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();started=time.perf_counter()
        bank=next(b for b in banks.values() if b['prefix']=='model.layers.0.mlp.gate_up_proj');layer=bank['layer'];packed=layer._tianshu_dense_w8_packed
        columns=torch.linspace(0,12287,32).long();cpu_w=packed.detach().cpu().contiguous()[columns];cpu_s=layer._tianshu_dense_w8_scale.detach().cpu()[columns]
        sizes=sorted({int(m) for m in cfg.compilation_config.cudagraph_capture_sizes if 128<=int(m)<=4096 and int(m)%128==0});assert sizes and 128 in sizes and 4096 in sizes
        dummy=torch.empty(1,device=packed.device,dtype=torch.int32)
        try:
            torch.set_num_threads(4)
            for m in sizes:
                reference_m=next(n for n in sorted(selected) if n>=m);config=dict(selected[reference_m])
                cpu_a=(((torch.arange(m,dtype=torch.int32)[:,None]%17)-8)*(((torch.arange(2048,dtype=torch.int32)[None,:]%13)-6))/8.).to(torch.bfloat16)
                cpu_a[0].zero_();a=cpu_a.to(packed.device);q=torch.empty((m,2048),device=a.device,dtype=torch.int8);s=torch.empty(m,device=a.device,dtype=torch.float32);out=torch.empty((m,12288),device=a.device,dtype=torch.bfloat16)
                for _ in range(2):kernel.quantize(a,q,s);compiled=kernel.mm(q,packed.T,s,layer._tianshu_dense_w8_scale,out,dummy,config)
                torch.cuda.synchronize();v=cpu_a.float();asc=v.abs().amax(1).clamp_min(1.e-8)/127.;aq=torch.round(v/asc[:,None]).clamp(-127,127).to(torch.int8)
                assert torch.equal(q.cpu(),aq) and torch.equal(s.cpu(),asc)
                rows=torch.tensor(sorted({0,1,m//2,m-1}),dtype=torch.long);exact=aq[rows].to(torch.int64)@cpu_w.to(torch.int64).T
                reference=exact.float()*asc[rows,None]*cpu_s[None,:];actual=out[rows.to(a.device)[:,None],columns.to(a.device)[None,:]].cpu()
                error=(actual.float()-reference).abs();assert bool(torch.isfinite(actual).all()) and not bool((error>(.02+.01*reference.abs())).any())
                warm_configs[m]=config;record['warmups'].append(dict(M=m,component_selection_M=reference_m,configuration=config,all_A_INT8_and_scales_exact=True,CPU_selected_real_W_INT64_to_BF16_passed=True,CPU_selected_outputs=actual.numel(),maximum_selected_error=float(error.max()),cache_hash=compiled.hash,binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared))
                del a,q,s,out,cpu_a,aq,asc,actual,reference,exact
        finally:
            torch.set_num_threads(previous_threads);record.update(warm_CPU_threads_before=previous_threads,warm_CPU_threads_after=torch.get_num_threads(),warm_CPU_threads_during=4)
        assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
        record.update(warm_CPU_GPU_RNG_unchanged=True,warm_all_declared_shapes_validated=True,warm_seconds=time.perf_counter()-started,warm_configs=warm_configs,stage='active',completed=True,activated_after_qualified_capture_and_own_warmup=True)
        torch.cuda.empty_cache();active=True;save();return value
    ModelRunnerFL.capture_model=capture;atexit.register(save);save()
