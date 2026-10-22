# SPDX-License-Identifier: Apache-2.0
import atexit,functools,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
from native_prefill_contract import use_native
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def install(worker,mode,directory):
    import torch
    import native_prefill_ops
    from native_prefill_fx import NativePrefillPass
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm_fl.worker.model_runner import ModelRunnerFL
    assert mode in ('control','candidate') and worker.vllm_config.speculative_config is None
    assert worker.vllm_config.model_config.model=='/workspace/MiniCPM5-2B'
    assert worker.vllm_config.scheduler_config.max_num_batched_tokens==4096 and worker.vllm_config.scheduler_config.max_num_seqs==64
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    kernel_path=EXP/'tianshu_native_int8_prefill_v2_20261022/kernel.py';quant_path=EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py'
    epilogue=load('_native_prefill_live_epilogue',kernel_path);quant=load('_native_prefill_live_quant',quant_path)
    component=json.loads((EXP/'tianshu_native_int8_prefill_v2_20261022/GPU_GATE_32g.json').read_text())
    holdout=json.loads((EXP/'tianshu_native_int8_mlp42_20261022/INDEPENDENT_CPU_GATE_32g.json').read_text());assert holdout['complete']
    expected={str(m):next(c for c in component['cases'] if c['M']==m)['variants']['NN_R1']['compiled']['binary_sha256'] for m in (512,4096)}
    cache_path=EXP/'tianshu_prefill_control_binary_all_20261022/QUALIFIED_MODEL_CACHE_32g.json';cache=json.loads(cache_path.read_text());assert cache['complete'] and os.environ['TRITON_CACHE_DIR']==cache['qualified_model_cache_directory']
    record=dict(installed=True,mode=mode,completed=False,phase='initializing',source=str(Path(__file__).resolve()),source_sha256=sha(__file__),kernel_source=str(kernel_path),kernel_sha256=sha(kernel_path),quantizer_source=str(quant_path),quantizer_sha256=sha(quant_path),cache_source=str(cache_path),cache_sha256=sha(cache_path),triton_cache_dir=os.environ['TRITON_CACHE_DIR'],supported_rows=[512,4096],no_new_weight_layout_or_payload=True,no_global_tensor_workspace=True,workspace_lifetime='private local tensors on current stream',routes={},native_kernels={},load_warmups=[],profile_windows=[],capture_completed=False)
    def save():
        path=directory/f'route_{os.getpid()}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
    def compiled_record(m,compiled):
        key=str(m)
        if key not in record['native_kernels']:
            binary=hashlib.sha256(compiled.kernel).hexdigest();assert binary==expected[key]
            record['native_kernels'][key]=dict(binary_sha256=binary,expected_binary_sha256=expected[key],component_binary_exact_match=True,cache_hash=compiled.hash,n_regs=compiled.n_regs,n_spills=compiled.n_spills)
    def compute(x,packed,scales):
        m=x.shape[0];q=torch.empty((m,2048),dtype=torch.int8,device=x.device);asc=torch.empty((m,),dtype=torch.float32,device=x.device);integer=torch.empty((m,12288),dtype=torch.int32,device=x.device);out=torch.empty((m,6144),dtype=torch.bfloat16,device=x.device)
        quant.quantize(x,q,asc);torch._int_mm(q,packed.T,out=integer);compiled=epilogue.epilogue(integer,asc,scales,out,out,1);compiled_record(m,compiled)
        return out
    def route(x,weight,packed,scales,bank_id):
        assert x.ndim==2 and x.dtype==torch.bfloat16 and x.shape[1]==2048 and x.stride(1)==1
        nn=sys.modules['_nn_silu_fx_NN_policy'];bank=nn.POINTERS[packed.data_ptr()];assert bank['bank_id']==bank_id
        m=x.shape[0];capturing=torch.cuda.is_current_stream_capturing();native=use_native(mode,m,x.shape[1],x.is_contiguous())
        implementation='native_A8W8_GateUp' if native else 'delegate_existing_range';key=str((bank_id,m,capturing,record['phase'],implementation))
        row=record['routes'].setdefault(key,dict(prefix=bank['prefix'],bank_id=bank_id,M=m,K=2048,N=12288,phase=record['phase'],capturing=capturing,implementation=implementation,input_stride=list(x.stride()),calls=0))
        row['calls']+=1
        if native:
            assert packed.shape==(12288,2048) and packed.stride()==(1,12288) and packed.T.is_contiguous()
            answer=compute(x,packed,scales);row.update(kernel_source=str(kernel_path),kernel_sha256=record['kernel_sha256'],quantizer_source=str(quant_path),quantizer_sha256=record['quantizer_sha256'],binary_sha256=record['native_kernels'][str(m)]['binary_sha256'],INT32_workspace_bytes=m*12288*4)
        else:answer=torch.ops.tianshu_m_range_prefill.fused_or_original.default(x,weight,packed,scales,bank_id)
        if bank['prefix']=='model.layers.41.mlp.gate_up_proj' and row['calls']&(row['calls']-1)==0:save()
        return answer
    native_prefill_ops.configure(route)
    cc=worker.vllm_config.compilation_config;parent_pass=cc.inductor_compile_config['post_grad_custom_post_pass'];cc.inductor_compile_config['post_grad_custom_post_pass']=NativePrefillPass(parent_pass,mode,directory/'fx_pass')
    original_process=UnquantizedLinearMethod.process_weights_after_loading;warmed=False
    def active_profiler():
        frame=inspect.currentframe()
        try:
            while frame:
                if frame.f_globals.get('__name__')=='vllm_fl.worker.model_runner' and frame.f_code.co_name=='load_model' and 'm' in frame.f_locals:
                    profiler=frame.f_locals['m'];assert hasattr(profiler,'initial_memory') and not hasattr(profiler,'final_memory')
                    return dict(active=True,source=str(Path(frame.f_code.co_filename).resolve()),sha256=sha(frame.f_code.co_filename))
                frame=frame.f_back
            raise AssertionError('native library warm outside actual DeviceMemoryProfiler')
        finally:del frame
    @functools.wraps(original_process)
    def process(method,layer):
        nonlocal warmed
        result=original_process(method,layer);nn=sys.modules.get('_nn_silu_fx_NN_policy')
        if warmed or nn is None or len(nn.LAYERS)!=42:return result
        if not hasattr(layer,'prefix') or not layer.prefix.endswith('.mlp.gate_up_proj'):return result
        evidence=active_profiler();warmed=True;record['phase']='load_warm';bank=next(l for l in nn.LAYERS.values() if l.prefix=='model.layers.0.mlp.gate_up_proj');before_cpu=torch.get_rng_state().clone();before_gpu=torch.cuda.get_rng_state().clone()
        for m in (512,4096):
            x=torch.zeros((m,2048),device=bank.weight.device,dtype=torch.bfloat16);out=compute(x,bank._tianshu_dense_w8_packed,bank._tianshu_dense_w8_scale);torch.cuda.synchronize();assert bool((out==0).all());record['load_warmups'].append(dict(M=m,zero_output=True,inside_DeviceMemoryProfiler=evidence,INT32_workspace_bytes=m*12288*4));del x,out
        assert torch.equal(torch.get_rng_state(),before_cpu) and torch.equal(torch.cuda.get_rng_state(),before_gpu);record['load_warm_RNG_unchanged']=True;record['phase']='before_profile';torch.cuda.empty_cache();save();return result
    UnquantizedLinearMethod.process_weights_after_loading=process
    original_profile=ModelRunnerFL.profile_run
    @functools.wraps(original_profile)
    def profile(owner,*args,**kwargs):
        previous=record['phase'];record['phase']='profile';frame=inspect.currentframe().f_back;context=None
        try:
            while frame:
                if frame.f_code.co_name=='determine_available_memory' and frame.f_globals.get('__name__')=='vllm_fl.worker.worker':
                    context=dict(source=str(Path(frame.f_code.co_filename).resolve()),sha256=sha(frame.f_code.co_filename),native_memory_context_active=('profile_result' in frame.f_locals),manual_KV_bytes=worker.cache_config.kv_cache_memory_bytes);break
                frame=frame.f_back
        finally:del frame
        item=dict(context=context,allocated_before=int(torch.cuda.memory_allocated()),peak_before=int(torch.cuda.max_memory_allocated()),native_calls_before=sum(r['calls'] for r in record['routes'].values() if r['phase']=='profile' and r['implementation']=='native_A8W8_GateUp'))
        try:return original_profile(owner,*args,**kwargs)
        finally:
            torch.cuda.synchronize();item.update(allocated_after=int(torch.cuda.memory_allocated()),peak_after=int(torch.cuda.max_memory_allocated()),native_calls_after=sum(r['calls'] for r in record['routes'].values() if r['phase']=='profile' and r['implementation']=='native_A8W8_GateUp'));record['profile_windows'].append(item);record['phase']=previous;save()
    ModelRunnerFL.profile_run=profile
    original_capture=ModelRunnerFL.capture_model
    @functools.wraps(original_capture)
    def capture(owner,*args,**kwargs):
        record['phase']='capture'
        try:answer=original_capture(owner,*args,**kwargs)
        except BaseException:save();raise
        record.update(phase='runtime',capture_completed=True,completed=True,model_memory_usage_bytes=int(owner.model_memory_usage));assert warmed
        if mode=='candidate':
            actual=[r for r in record['routes'].values() if r['phase']=='profile' and r['M']==4096 and r['implementation']=='native_A8W8_GateUp'];assert len({r['bank_id'] for r in actual})==42
            assert any(w['context'] and w['context']['native_memory_context_active'] and not w['context']['manual_KV_bytes'] and w['native_calls_after']-w['native_calls_before']>=42 for w in record['profile_windows'])
        save();return answer
    ModelRunnerFL.capture_model=capture;atexit.register(save);save()
