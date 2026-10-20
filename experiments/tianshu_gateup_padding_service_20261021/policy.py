# SPDX-License-Identifier: Apache-2.0
"""Selective W8A8 with loader-time accounting and opaque Torch custom-op dispatch."""
import atexit,functools,hashlib,importlib.util,inspect,json,os,re,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
COMPONENT=BASE.parent/'tianshu_dense_w8a8_20261020'
KERNEL_SHA='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152'
PLAN=json.loads((COMPONENT/'MODEL_INTEGRATION_PLAN_32g.json').read_text())
CONFIGS={(r['case'],r['M']):r['configuration'] for r in PLAN['selected_configurations']}

def role_for(kind,n,k):
    if kind=='linear' and (n,k)==(12288,2048):return 'gateup'
    if kind=='lm_head' and (n,k)==(130560,2048):return 'lm_head'
    return None

def config_for(role,m):return CONFIGS.get((role,m))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def install(worker,mode,directory):
    assert mode in ('control','candidate')
    cold_mode=os.environ["TIANSHU_COLD_START_MODE"];assert cold_mode in ("native","cpu4","gpu4")
    precision_mode=os.environ["TIANSHU_LM_PRECISION_MODE"];assert precision_mode in ("control","candidate")
    padding_mode=os.environ['TIANSHU_TILE_PADDING_MODE'];assert padding_mode in ('control','candidate')
    local_configs=dict(CONFIGS)
    if precision_mode=="candidate":local_configs.pop(("lm_head",8))
    if padding_mode=='candidate':
        for m in range(8,65):local_configs[('gateup',m)]=CONFIGS[('gateup',8 if m<=32 else 64)]
    assert os.environ.get('VLLM_DISABLE_COMPILE_CACHE')=='1'
    assert worker.vllm_config.speculative_config is None
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod,ParallelLMHead
    from vllm_fl.worker.model_runner import ModelRunnerFL
    assert not getattr(ModelRunnerFL,'_tianshu_dense_int8_installed',False)
    assert digest(COMPONENT/'kernel.py')==KERNEL_SHA
    gate=json.loads((COMPONENT/'GPU_GATE_32g.json').read_text())
    assert gate['complete'] and gate['independently_CPU_rechecked'] and gate['compiled_cases']==113
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('_tianshu_dense_integer_service_kernel',COMPONENT/'kernel.py')
    kernel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernel;spec.loader.exec_module(kernel)
    pad_path=BASE.parent/'tianshu_dense_tile_padding_20261021/kernel_padded.py'
    pad_gate=json.loads((pad_path.parent/'GPU_GATE_32g.json').read_text());assert pad_gate['complete'] and pad_gate['independently_CPU_rechecked']
    assert digest(pad_path)==pad_gate['frozen_sources'][str(pad_path)]
    pad_spec=importlib.util.spec_from_file_location('_tianshu_gateup_padding_actual_kernel',pad_path)
    pad_kernel=importlib.util.module_from_spec(pad_spec);sys.modules[pad_spec.name]=pad_kernel;pad_spec.loader.exec_module(pad_kernel)
    pad_sha=digest(pad_path)
    pack_path=BASE.parent/"tianshu_w8_gpu_packing_20261020/packing.py"
    pack_spec=importlib.util.spec_from_file_location("_tianshu_cold_start_gpu_packing",pack_path)
    packer=importlib.util.module_from_spec(pack_spec);sys.modules[pack_spec.name]=packer;pack_spec.loader.exec_module(packer)
    banks={};dummy=None;active_runner=None
    originals={kind:(cls.process_weights_after_loading,cls.apply) for kind,cls in (('linear',UnquantizedLinearMethod),('embedding',UnquantizedEmbeddingMethod))}
    native_paths={str(Path(inspect.getsourcefile(fn)).resolve()):digest(inspect.getsourcefile(fn)) for pair in originals.values() for fn in pair}
    record=dict(installed=True,mode=mode,startup_mode=cold_mode,cpu_num_threads_at_install=torch.get_num_threads(),compile_cache_disabled=True,policy_source=str(Path(__file__).resolve()),policy_sha256=digest(__file__),
        kernel_source=str(COMPONENT/'kernel.py'),kernel_sha256=KERNEL_SHA,native_method_sources=native_paths,
        supported_M=[8,64],supported_roles=['gateup','lm_head'],configs={f'{role}:{m}':cfg for (role,m),cfg in local_configs.items()},
        banks=[],visited=[],launches={},load_complete=False,packed_payload_bytes=0,original_BF16_weights_retained=True,
        checkpoint_never_modified=True,embedding_not_packed=True,GateUp_unsupported_M_uses_qualified_BF16_mm=(os.environ['TIANSHU_PREFILL_LINEAR_MODE']=='control'),warmup_retains_qualified_BF16_mm=True,packing_inside_actual_active_DeviceMemoryProfiler=True,
        numerical_approximation_changes_model=True,RNG_not_read_or_modified=True,Python_launch_counts_are_not_Graph_replays=True)
    record.update(lm_precision_mode=precision_mode,BF16_lm_launches={},supported_M_by_role={"gateup":[8,64],"lm_head":([64] if precision_mode=="candidate" else [8,64])})
    record.update(padding_mode=padding_mode,padding_kernel_source=str(pad_path),padding_kernel_sha256=pad_sha,
        supported_M=sorted({m for role,m in local_configs}),supported_M_by_role={role:sorted(m for r,m in local_configs if r==role) for role in ("gateup","lm_head")},
        fresh_buffers_per_custom_op=True,returned_valid_rows_only=True,LM_precision_policy_unchanged=True)
    def save():
        path=directory/f'bindings_{os.getpid()}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
    def profiler_evidence():
        frame=inspect.currentframe().f_back
        try:
            while frame is not None:
                if frame.f_globals.get('__name__')=='vllm_fl.worker.model_runner' and frame.f_code.co_name=='load_model' and 'm' in frame.f_locals:
                    profiler=frame.f_locals['m']
                    assert id(frame.f_locals['self'])==active_runner
                    assert hasattr(profiler,'initial_memory') and not hasattr(profiler,'final_memory') and not hasattr(profiler,'consumed_memory')
                    return dict(source=str(Path(frame.f_code.co_filename).resolve()),sha256=digest(frame.f_code.co_filename),
                        function='load_model',initial_device_memory=int(profiler.initial_memory),profiler_not_exited=True)
                frame=frame.f_back
            raise AssertionError('Weight packing outside the actual FL model memory-profiler context')
        finally:del frame
    def prepare_layer(method,layer,kind):
        nonlocal dummy
        originals[kind][0](method,layer)
        weight=layer.weight
        if weight.ndim!=2:return
        actual_kind='lm_head' if isinstance(layer,ParallelLMHead) else ('linear' if kind=='linear' else 'embedding')
        role=role_for(actual_kind,*weight.shape)
        if role is None:return
        assert active_runner is not None and weight.dtype==torch.bfloat16 and weight.is_cuda and weight.is_contiguous()
        prefix=getattr(layer,'prefix',None) or ('lm_head' if role=='lm_head' else '')
        if role=='gateup':assert re.fullmatch(r'model\.layers\.\d+\.mlp\.gate_up_proj',prefix),prefix
        proof=profiler_evidence();version=int(weight._version)
        record['visited'].append(dict(role=role,prefix=prefix,method_class=type(method).__module__+'.'+type(method).__name__,shape=list(weight.shape),inside_profiler=proof,original_weight_version=version))
        if mode=='control':save();return
        assert not hasattr(layer,'_tianshu_dense_w8_bank')
        start=time.perf_counter();cpu=weight.detach().cpu().contiguous();assert bool(torch.isfinite(cpu).all())
        n,k=cpu.shape
        if cold_mode=="gpu4":
            q=torch.empty((n,k),device=weight.device,dtype=torch.int8);s=torch.empty(n,device=weight.device,dtype=torch.float32)
            packer.pack_into(weight,q,s,kernel);packed=q.cpu();scales=s.cpu()
        else:
            packed=torch.empty((n,k),dtype=torch.int8);scales=torch.empty(n,dtype=torch.float32)
            for lo in range(0,n,8192):
                block=cpu[lo:lo+8192].float();scale=block.abs().amax(1).clamp_min(1.e-8)/127.
                packed[lo:lo+len(block)]=torch.round(block/scale[:,None]).clamp(-127,127).to(torch.int8);scales[lo:lo+len(block)]=scale
            q,s=packed.to(weight.device),scales.to(weight.device)
        if dummy is None:dummy=torch.empty(1,device=weight.device,dtype=torch.int32)
        bank_id=len(banks);layer.register_buffer('_tianshu_dense_w8_packed',q,persistent=False)
        layer.register_buffer('_tianshu_dense_w8_scale',s,persistent=False);layer._tianshu_dense_w8_bank=bank_id
        banks[bank_id]=dict(layer=layer,method=method,kind=kind,role=role,prefix=prefix)
        assert int(weight._version)==version
        row=dict(bank_id=bank_id,role=role,prefix=prefix,method_class=type(method).__module__+'.'+type(method).__name__,shape=[n,k],inside_profiler=proof,original_weight_unchanged=True,
            original_BF16_sha256=hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest(),
            packed_sha256=hashlib.sha256(memoryview(packed.view(torch.uint8).numpy())).hexdigest(),
            scales_sha256=hashlib.sha256(memoryview(scales.view(torch.uint8).numpy())).hexdigest(),
            packed_bytes=packed.numel()+scales.numel()*4,packing_method=("GPU" if cold_mode=="gpu4" else "CPU"),packing_CPU_threads=torch.get_num_threads(),packing_seconds=time.perf_counter()-start)
        record['banks'].append(row);record['packed_payload_bytes']+=row['packed_bytes'];save()
    def native_lm_fallback(bank,x):
        assert bank["role"]=="lm_head"
        capturing=torch.cuda.is_current_stream_capturing();m=x.shape[0]
        key=str((bank["prefix"],m,capturing))
        row=record["BF16_lm_launches"].setdefault(key,dict(prefix=bank["prefix"],M=m,N=bank["layer"].weight.shape[0],actually_inside_capture=capturing,implementation="original_embedding_method_BF16",calls=0))
        output=originals[bank["kind"]][1](bank["method"],bank["layer"],x,None)
        assert x.dtype==bank["layer"].weight.dtype==output.dtype==torch.bfloat16
        row.update(input_dtype=str(x.dtype),weight_dtype=str(bank["layer"].weight.dtype),output_dtype=str(output.dtype))
        row["calls"]+=1
        if not capturing and row["calls"]&(row["calls"]-1)==0:save()
        return output
    @torch.library.custom_op('tianshu_dense_w8a8::linear',mutates_args=(),device_types='cuda')
    def int8_linear(x:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
        bank=banks[bank_id];m,k=x.shape;role=bank['role'];cfg=local_configs.get((role,m))
        if cfg is None:
            if role=='gateup':
                from flag_gems import mm
                return route(x,weight,bank,mm)
            return native_lm_fallback(bank,x)
        assert x.dtype==torch.bfloat16 and x.stride(-1)==1 and weight.shape[1]==k
        selected=padding_mode=="candidate" and role=="gateup"
        allocated_m=(32 if m<=32 else 64) if selected else m
        q=torch.empty((allocated_m,k),device=x.device,dtype=torch.int8);ascale=torch.empty(allocated_m,device=x.device,dtype=torch.float32)
        out=torch.empty((allocated_m,weight.shape[0]),device=x.device,dtype=torch.bfloat16)
        implementation=pad_kernel if selected else kernel
        quant=implementation.quantize(x,q,ascale);gemm=implementation.mm(q,packed,ascale,scales,out,dummy,cfg)
        capturing=torch.cuda.is_current_stream_capturing()
        module=sys.modules.get('_tianshu_long_segments_actual_graph_trace');trace=getattr(module,'STATE',{})
        graph_key=trace.get('active_graph');key=str((bank_id,m,capturing,graph_key))
        row=record['launches'].setdefault(key,dict(bank_id=bank_id,role=role,prefix=bank['prefix'],M=m,K=k,N=weight.shape[0],
            actually_inside_capture=capturing,graph_key=graph_key,configuration=cfg,kernel_sha256=KERNEL_SHA,
            n_regs=getattr(gemm,'n_regs',None),n_spills=getattr(gemm,'n_spills',None),calls=0))
        row.update(allocated_M=allocated_m,padding_selected=selected,actually_has_padding=allocated_m!=m,
            quantizer_source_sha256=(pad_sha if selected and allocated_m!=m else KERNEL_SHA),quantizer_cache_hash=quant.hash,
            allocated_q_shape=list(q.shape),allocated_scale_shape=list(ascale.shape),allocated_output_shape=list(out.shape),
            returned_output_shape=[m,weight.shape[0]],actual_ephemeral_buffer_bytes=q.numel()+4*ascale.numel()+2*out.numel(),
            matrix_pointer_alignment_mod256=[t.data_ptr()%256 for t in (q,packed,ascale,scales,out,dummy)])
        row['calls']+=1
        if not capturing and (row['calls'] & (row['calls']-1))==0:save()
        return out[:m] if selected else out
    @int8_linear.register_fake
    def int8_linear_fake(x,weight,packed,scales,bank_id):return torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=x.dtype)
    def make_process(kind):
        def process(method,layer):return prepare_layer(method,layer,kind)
        return process
    def make_apply(kind):
        def apply(method,layer,x,bias=None):
            if bias is not None or not hasattr(layer,'_tianshu_dense_w8_bank') or x.ndim!=2 or x.dtype!=torch.bfloat16 or x.stride(-1)!=1:
                return originals[kind][1](method,layer,x,bias)
            return int8_linear(x,layer.weight,layer._tianshu_dense_w8_packed,layer._tianshu_dense_w8_scale,layer._tianshu_dense_w8_bank)
        return apply
    for kind,cls in (('linear',UnquantizedLinearMethod),('embedding',UnquantizedEmbeddingMethod)):
        cls.process_weights_after_loading=make_process(kind);cls.apply=make_apply(kind)
    from prefill_route import install_route
    route=install_route(ModelRunnerFL,banks,record,save)
    original_load=ModelRunnerFL.load_model
    @functools.wraps(original_load)
    def load_model(owner,*args,**kwargs):
        nonlocal active_runner
        assert active_runner is None and not record['load_complete'];active_runner=id(owner)
        started=time.perf_counter();previous_threads=torch.get_num_threads();record["cpu_threads_before_load"]=previous_threads
        try:
            if cold_mode!="native":torch.set_num_threads(4)
            record["cpu_threads_during_load"]=torch.get_num_threads()
            value=original_load(owner,*args,**kwargs)
        finally:
            active_runner=None
            if torch.get_num_threads()!=previous_threads:torch.set_num_threads(previous_threads)
            record["cpu_threads_after_load"]=torch.get_num_threads();record["load_model_seconds"]=time.perf_counter()-started;save()
        visited=record['visited'];assert len(visited)==43 and sum(r['role']=='gateup' for r in visited)==42 and sum(r['role']=='lm_head' for r in visited)==1
        assert len(banks)==(43 if mode=='candidate' else 0)
        assert record['packed_payload_bytes']==(1326938112 if mode=='candidate' else 0)
        record.update(load_complete=True,model_memory_usage_bytes=int(owner.model_memory_usage),
            parameter_device_bytes=sum(p.numel()*p.element_size() for p in owner.model.parameters() if p.is_cuda),
            torch_allocated_after_load=int(torch.cuda.memory_allocated()),actual_packed_banks=len(banks))
        assert record['model_memory_usage_bytes']+16*1024**2>=record['parameter_device_bytes']+record['packed_payload_bytes']
        save();return value
    ModelRunnerFL.load_model=load_model;ModelRunnerFL._tianshu_dense_int8_installed=True
    atexit.register(save);save()
