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
    banks={};dummy=None;active_runner=None
    originals={kind:(cls.process_weights_after_loading,cls.apply) for kind,cls in (('linear',UnquantizedLinearMethod),('embedding',UnquantizedEmbeddingMethod))}
    native_paths={str(Path(inspect.getsourcefile(fn)).resolve()):digest(inspect.getsourcefile(fn)) for pair in originals.values() for fn in pair}
    record=dict(installed=True,mode=mode,policy_source=str(Path(__file__).resolve()),policy_sha256=digest(__file__),
        kernel_source=str(COMPONENT/'kernel.py'),kernel_sha256=KERNEL_SHA,native_method_sources=native_paths,
        supported_M=[8,64],supported_roles=['gateup','lm_head'],configs={f'{role}:{m}':cfg for (role,m),cfg in CONFIGS.items()},
        banks=[],visited=[],launches={},load_complete=False,packed_payload_bytes=0,original_BF16_weights_retained=True,
        checkpoint_never_modified=True,embedding_not_packed=True,packing_inside_actual_active_DeviceMemoryProfiler=True,
        numerical_approximation_changes_model=True,RNG_not_read_or_modified=True,Python_launch_counts_are_not_Graph_replays=True)
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
        record['visited'].append(dict(role=role,prefix=prefix,shape=list(weight.shape),inside_profiler=proof,original_weight_version=version))
        if mode=='control':save();return
        assert not hasattr(layer,'_tianshu_dense_w8_bank')
        start=time.perf_counter();cpu=weight.detach().cpu().contiguous();assert bool(torch.isfinite(cpu).all())
        n,k=cpu.shape;packed=torch.empty((n,k),dtype=torch.int8);scales=torch.empty(n,dtype=torch.float32)
        for lo in range(0,n,8192):
            block=cpu[lo:lo+8192].float();scale=block.abs().amax(1).clamp_min(1.e-8)/127.
            packed[lo:lo+len(block)]=torch.round(block/scale[:,None]).clamp(-127,127).to(torch.int8);scales[lo:lo+len(block)]=scale
        q,s=packed.to(weight.device),scales.to(weight.device)
        if dummy is None:dummy=torch.empty(1,device=weight.device,dtype=torch.int32)
        bank_id=len(banks);layer.register_buffer('_tianshu_dense_w8_packed',q,persistent=False)
        layer.register_buffer('_tianshu_dense_w8_scale',s,persistent=False);layer._tianshu_dense_w8_bank=bank_id
        banks[bank_id]=dict(layer=layer,method=method,kind=kind,role=role,prefix=prefix)
        assert int(weight._version)==version
        row=dict(bank_id=bank_id,role=role,prefix=prefix,shape=[n,k],inside_profiler=proof,original_weight_unchanged=True,
            original_BF16_sha256=hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest(),
            packed_sha256=hashlib.sha256(memoryview(packed.view(torch.uint8).numpy())).hexdigest(),
            scales_sha256=hashlib.sha256(memoryview(scales.view(torch.uint8).numpy())).hexdigest(),
            packed_bytes=packed.numel()+scales.numel()*4,packing_seconds=time.perf_counter()-start)
        record['banks'].append(row);record['packed_payload_bytes']+=row['packed_bytes'];save()
    @torch.library.custom_op('tianshu_dense_w8a8::linear',mutates_args=(),device_types='cuda')
    def int8_linear(x:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
        bank=banks[bank_id];m,k=x.shape;role=bank['role'];cfg=config_for(role,m)
        if cfg is None:return originals[bank['kind']][1](bank['method'],bank['layer'],x,None)
        assert x.dtype==torch.bfloat16 and x.stride(-1)==1 and weight.shape[1]==k
        q=torch.empty((m,k),device=x.device,dtype=torch.int8);ascale=torch.empty(m,device=x.device,dtype=torch.float32)
        out=torch.empty((m,weight.shape[0]),device=x.device,dtype=torch.bfloat16)
        quant=kernel.quantize(x,q,ascale);gemm=kernel.mm(q,packed,ascale,scales,out,dummy,cfg)
        capturing=torch.cuda.is_current_stream_capturing()
        module=sys.modules.get('_tianshu_long_segments_actual_graph_trace');trace=getattr(module,'STATE',{})
        graph_key=trace.get('active_graph');key=str((bank_id,m,capturing,graph_key))
        row=record['launches'].setdefault(key,dict(bank_id=bank_id,role=role,prefix=bank['prefix'],M=m,K=k,N=weight.shape[0],
            actually_inside_capture=capturing,graph_key=graph_key,configuration=cfg,kernel_sha256=KERNEL_SHA,
            n_regs=getattr(gemm,'n_regs',None),n_spills=getattr(gemm,'n_spills',None),calls=0))
        row['calls']+=1
        if not capturing and (row['calls'] & (row['calls']-1))==0:save()
        return out
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
    original_load=ModelRunnerFL.load_model
    @functools.wraps(original_load)
    def load_model(owner,*args,**kwargs):
        nonlocal active_runner
        assert active_runner is None and not record['load_complete'];active_runner=id(owner)
        try:value=original_load(owner,*args,**kwargs)
        finally:active_runner=None
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
