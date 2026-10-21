# SPDX-License-Identifier: Apache-2.0
"""Add QKV prefill A8/W8 under the actual native model-load memory profiler."""
import atexit,functools,hashlib,importlib.util,inspect,json,os,re,sys,time
from pathlib import Path
from qkv_prefill_weight_pack import pack,tensor_sha
from qkv_prefill_routing import configuration,eligible
BASE=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()

def install(worker,mode,directory):
    assert mode in ('control','candidate')
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm_fl.worker.model_runner import ModelRunnerFL
    cfg=worker.vllm_config;hf=cfg.model_config.hf_config
    assert cfg.model_config.model=='/workspace/MiniCPM5-2B' and cfg.model_config.quantization is None
    assert (hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads)==(2048,42,16,2)
    assert cfg.speculative_config is None and cfg.lora_config is None
    assert cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.async_scheduling
    assert cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==cfg.parallel_config.data_parallel_size==1
    assert os.environ['TIANSHU_LM_PRECISION_MODE']=='candidate' and os.environ['VLLM_DISABLE_COMPILE_CACHE']=='1'
    selection=json.loads((BASE/'selected_plan.json').read_text());kernel_path=Path(selection['kernel_source'])
    assert sha(kernel_path)==selection['kernel_sha256'] and sha(selection['component_gate'])==selection['component_gate_sha256']
    gate=json.loads(Path(selection['component_gate']).read_text())
    assert gate['complete'] and gate['independently_CPU_rechecked'] and gate['original_GPU_run_remains_failed']
    assert gate['partial_scope']=='completed_QKV_only' and all(sha(p)==v for p,v in gate['frozen_sources'].items())
    selected={row['M']:row['configuration'] for row in selection['configurations']};assert set(selected)=={128,256,512,1024,2048,4096}
    manifest=json.loads((BASE/'CHECKPOINT_SIGNATURES_32g.json').read_text());assert manifest['complete'] and len(manifest['banks'])==42
    expected={row['prefix']:row for row in manifest['banks']}
    spec=importlib.util.spec_from_file_location('_qkv_prefill_integer_kernel',kernel_path)
    kernel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernel;spec.loader.exec_module(kernel)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    original_process=UnquantizedLinearMethod.process_weights_after_loading
    original_apply=UnquantizedLinearMethod.apply;original_capture=ModelRunnerFL.capture_model
    record=dict(installed=True,mode=mode,completed=False,stage='before_capture',banks=[],routes={},warmups=[],warm_configs={},
        policy_source=str(BASE/'policy.py'),policy_sha256=sha(__file__),kernel_source=str(kernel_path),kernel_sha256=sha(kernel_path),
        selection_sha256=sha(BASE/'selected_plan.json'),checkpoint_signatures_sha256=sha(BASE/'CHECKPOINT_SIGNATURES_32g.json'),
        QKV_prefill_only=True,decode_unchanged=True,O_not_intercepted=True,original_BF16_weights_retained=True,
        packing_inside_actual_active_DeviceMemoryProfiler=True,unknown_M_or_layout_falls_back_without_new_INT8_JIT=True,
        Python_counts_are_not_Graph_replays=True,CPU_warm_reference_scope='All A quantization and selected INT64 outputs; not full warm output CPU recomputation')
    banks={};warm_configs={};active=False;dummy=None
    def save():
        path=directory/f'route_{os.getpid()}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
    def emit(row):
        row['calls']+=1
        if row['prefix']=='model.layers.41.self_attn.qkv_proj' and row['calls']&(row['calls']-1)==0:save()
    def profiler():
        frame=inspect.currentframe().f_back
        try:
            while frame:
                if frame.f_globals.get('__name__')=='vllm_fl.worker.model_runner' and frame.f_code.co_name=='load_model' and 'm' in frame.f_locals:
                    value=frame.f_locals['m'];assert hasattr(value,'initial_memory') and not hasattr(value,'final_memory')
                    return dict(active=True,source=str(Path(frame.f_code.co_filename).resolve()),sha256=sha(frame.f_code.co_filename),initial_memory=int(value.initial_memory))
                frame=frame.f_back
            raise AssertionError('No actual active native DeviceMemoryProfiler')
        finally:del frame
    @functools.wraps(original_process)
    def process(method,layer):
        value=original_process(method,layer)
        prefix=getattr(layer,'prefix','')
        if not re.fullmatch(r'model\.layers\.\d+\.self_attn\.qkv_proj',prefix):return value
        assert tuple(layer.weight.shape)==(2560,2048) and layer.weight.dtype==torch.bfloat16
        bank_id=int(prefix.split('.')[2]);assert bank_id not in banks and 0<=bank_id<42
        proof=profiler();weight=layer.weight;version=int(weight._version);pointer=weight.data_ptr();started=time.perf_counter()
        cpu=weight.detach().cpu().contiguous();signature=expected[prefix]
        assert tensor_sha(cpu)==signature['original_BF16_sha256']
        packed,scales=weight,weight;payload=0
        if mode=='candidate':
            q,s=pack(cpu)
            assert tensor_sha(q)==signature['packed_logical_NK_sha256'] and tensor_sha(s)==signature['scales_sha256']
            packed=q.T.contiguous().to(weight.device);scales=s.to(weight.device)
            assert tensor_sha(packed.detach().cpu())==signature['packed_physical_KN_sha256'] and tensor_sha(scales.detach().cpu())==signature['scales_sha256']
            payload=packed.numel()*packed.element_size()+scales.numel()*scales.element_size();assert payload==signature['payload_bytes']
        assert weight.data_ptr()==pointer and int(weight._version)==version and tensor_sha(weight.detach().cpu())==signature['original_BF16_sha256']
        layer._tianshu_qkv_prefill_bank=bank_id;layer._tianshu_qkv_prefill_packed=packed;layer._tianshu_qkv_prefill_scales=scales
        banks[bank_id]=dict(layer=layer,method=method,prefix=prefix)
        record['banks'].append(dict(bank_id=bank_id,prefix=prefix,signature=signature,payload_bytes=payload,inside_profiler=proof,
            original_weight_unchanged=True,original_weight_version=version,original_weight_pointer=pointer,
            actual_packed_shape=list(packed.shape),actual_packed_stride=list(packed.stride()),actual_packed_dtype=str(packed.dtype),
            control_placeholder_alias_original=(mode=='control'),packing_seconds=time.perf_counter()-started))
        save();return value
    UnquantizedLinearMethod.process_weights_after_loading=process
    @torch.library.custom_op('tianshu_qkv_prefill_int8::linear',mutates_args=(),device_types='cuda')
    def linear(x:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
        bank=banks[bank_id];m=x.shape[0];capturing=torch.cuda.is_current_stream_capturing()
        use=eligible(mode,active,x,warm_configs);implementation='prefill_A8W8' if use else 'delegate_qualified'
        key=str((bank_id,m,record['stage'],capturing,implementation))
        row=record['routes'].setdefault(key,dict(prefix=bank['prefix'],M=m,stage=record['stage'],actually_inside_capture=capturing,
            implementation=implementation,input_stride=list(x.stride()),calls=0))
        if not use:
            output=original_apply(bank['method'],bank['layer'],x,None);emit(row);return output
        assert x.dtype==torch.bfloat16 and packed.shape==(2048,2560) and packed.is_contiguous() and scales.shape==(2560,)
        config=warm_configs[m];q=torch.empty((m,2048),device=x.device,dtype=torch.int8);s=torch.empty(m,device=x.device,dtype=torch.float32)
        output=torch.empty((m,2560),device=x.device,dtype=torch.bfloat16)
        kernel.quantize(x,q,s);compiled=kernel.mm(q,packed,s,scales,output,dummy,config)
        if 'kernel_sha256' not in row:
            row.update(configuration=config,kernel_source=str(kernel_path),kernel_sha256=sha(kernel_path),cache_hash=compiled.hash,
                binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared)
        emit(row);return output
    @linear.register_fake
    def fake(x,weight,packed,scales,bank_id):return torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=x.dtype)
    @functools.wraps(original_apply)
    def apply(method,layer,x,bias=None):
        if bias is not None or not hasattr(layer,'_tianshu_qkv_prefill_bank') or x.ndim!=2 or x.dtype!=torch.bfloat16 or x.stride(-1)!=1:return original_apply(method,layer,x,bias)
        return linear(x,layer.weight,layer._tianshu_qkv_prefill_packed,layer._tianshu_qkv_prefill_scales,int(layer._tianshu_qkv_prefill_bank))
    UnquantizedLinearMethod.apply=apply
    @functools.wraps(original_capture)
    def capture(owner,*args,**kwargs):
        nonlocal active,dummy
        assert not active and not record['completed']
        value=original_capture(owner,*args,**kwargs);assert len(banks)==42 and set(banks)==set(range(42))
        for m in (8,64):
            routes=[r for r in record['routes'].values() if r['M']==m and r['actually_inside_capture']]
            assert len({r['prefix'] for r in routes})==42 and all(r['implementation']=='delegate_qualified' for r in routes)
        record.update(stage='warming',after_original_capture=True,parent_prefill_int8_remains_candidate=True);save()
        sizes=sorted({int(m) for m in cfg.compilation_config.cudagraph_capture_sizes if 128<=int(m)<=4096 and int(m)%128==0})
        assert sizes and 128 in sizes and 4096 in sizes;record['native_prefill_capture_sizes']=sizes
        previous_threads=torch.get_num_threads();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();started=time.perf_counter()
        if mode=='candidate':
            bank=banks[0];layer=bank['layer'];packed=layer._tianshu_qkv_prefill_packed;scales=layer._tianshu_qkv_prefill_scales
            columns=torch.linspace(0,2559,32).long();cpu_w=packed.detach().cpu().T.contiguous()[columns];cpu_s=scales.detach().cpu()[columns]
            dummy=torch.empty(1,device=packed.device,dtype=torch.int32)
            try:
                torch.set_num_threads(4)
                for m in sizes:
                    reference_m,config=configuration(m,selected)
                    cpu_a=(((torch.arange(m,dtype=torch.int32)[:,None]%17)-8)*(((torch.arange(2048,dtype=torch.int32)[None,:]%13)-6))/8.).to(torch.bfloat16)
                    cpu_a[0].zero_();a=cpu_a.to(packed.device)
                    q=torch.empty((m,2048),device=a.device,dtype=torch.int8);s=torch.empty(m,device=a.device,dtype=torch.float32);out=torch.empty((m,2560),device=a.device,dtype=torch.bfloat16)
                    for _ in range(2):kernel.quantize(a,q,s);compiled=kernel.mm(q,packed,s,scales,out,dummy,config)
                    torch.cuda.synchronize();v=cpu_a.float();asc=v.abs().amax(1).clamp_min(1.e-8)/127.;aq=torch.round(v/asc[:,None]).clamp(-127,127).to(torch.int8)
                    assert torch.equal(q.cpu(),aq) and torch.equal(s.cpu(),asc)
                    rows=torch.tensor(sorted({0,1,m//2,m-1}),dtype=torch.long);exact=aq[rows].to(torch.int64)@cpu_w.to(torch.int64).T
                    reference=exact.float()*asc[rows,None]*cpu_s[None,:];actual=out[rows.to(a.device)[:,None],columns.to(a.device)[None,:]].cpu()
                    error=(actual.float()-reference).abs();assert bool(torch.isfinite(actual).all()) and not bool((error>(.02+.01*reference.abs())).any())
                    warm_configs[m]=config
                    record['warmups'].append(dict(M=m,component_selection_M=reference_m,configuration=config,all_A_INT8_and_scales_exact=True,
                        CPU_selected_real_W_INT64_to_BF16_passed=True,CPU_selected_outputs=actual.numel(),maximum_selected_error=float(error.max()),
                        cache_hash=compiled.hash,binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared))
                    del a,q,s,out,cpu_a,aq,asc,actual,reference,exact
            finally:
                torch.set_num_threads(previous_threads)
        assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
        record.update(warm_CPU_threads_before=previous_threads,warm_CPU_threads_after=torch.get_num_threads(),warm_CPU_threads_during=(4 if mode=='candidate' else previous_threads),
            warm_CPU_GPU_RNG_unchanged=True,warm_all_declared_shapes_validated=(set(warm_configs)==set(sizes) if mode=='candidate' else True),
            warm_seconds=time.perf_counter()-started,warm_configs=warm_configs,stage='active',completed=True,
            activated_after_qualified_capture_and_own_warmup=True,actual_model_memory_usage_bytes=int(owner.model_memory_usage),
            actual_extra_packed_payload_bytes=sum(r['payload_bytes'] for r in record['banks']))
        assert record['actual_extra_packed_payload_bytes']==(220631040 if mode=='candidate' else 0)
        torch.cuda.empty_cache();active=True;save();return value
    ModelRunnerFL.capture_model=capture
    atexit.register(save);save()
