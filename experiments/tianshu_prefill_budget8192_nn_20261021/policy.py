# SPDX-License-Identifier: Apache-2.0
"""GateUp-only physical W layout, same logical bank and payload, native LM route."""
import atexit,functools,hashlib,importlib.util,inspect,json,os,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
LAYOUT=BASE.parent/'tianshu_int8_weight_nn_20261021'
RESULT=BASE.parent/'tianshu_int8_nn_gateup_tiles_results_20261021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install(worker,mode,directory):
    assert mode in ('control','candidate')
    import torch
    from vllm.platforms import current_platform
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm_fl.worker.model_runner import ModelRunnerFL
    cfg=worker.vllm_config;hf=cfg.model_config.hf_config
    assert current_platform.vendor_name=='iluvatar' and cfg.model_config.model=='/workspace/MiniCPM5-2B'
    assert (hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads,hf.vocab_size)==(2048,42,16,2,130560)
    assert cfg.speculative_config is None and cfg.lora_config is None and cfg.model_config.quantization is None
    assert cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==cfg.parallel_config.data_parallel_size==1
    assert cfg.scheduler_config.max_num_batched_tokens=={'control':4096,'candidate':8192}[os.environ['TIANSHU_DENSE_BUDGET_MODE']] and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.async_scheduling
    assert os.environ['TIANSHU_LM_PRECISION_MODE']=='candidate' and os.environ['TIANSHU_COLD_START_MODE']=='gpu4' and os.environ['VLLM_DISABLE_COMPILE_CACHE']=='1'
    gate=json.loads((RESULT/'GPU_GATE_32g.json').read_text());assert gate['complete'] and gate['passed_cases']==34 and not gate['compile_failures']
    assert all(sha(p)==v for p,v in gate['frozen_sources'].items())
    configs={r['M']:r['NN_configuration'] for r in gate['best_by_M']};assert set(configs)=={8,64}
    kernel=sys.modules['_tianshu_dense_integer_service_kernel'];original_mm=kernel.mm;original_process=UnquantizedLinearMethod.process_weights_after_loading
    assert Path(inspect.getsourcefile(original_mm)).resolve()==BASE.parent/'tianshu_dense_w8a8_20261020/kernel.py'
    spec=importlib.util.spec_from_file_location('_gateup_NN_service_kernel',LAYOUT/'kernel_nn.py');nn=importlib.util.module_from_spec(spec);sys.modules[spec.name]=nn;spec.loader.exec_module(nn)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    record=dict(installed=True,mode=mode,completed=False,banks=[],routes={},selected_configurations=configs,policy_source=str(BASE/'policy.py'),policy_sha256=sha(__file__),NN_source=str(LAYOUT/'kernel_nn.py'),NN_sha256=sha(LAYOUT/'kernel_nn.py'),NT_source=str(inspect.getsourcefile(original_mm)),NT_sha256=sha(inspect.getsourcefile(original_mm)),
        original_parameter_and_checkpoint_unchanged=True,logical_packed_banks_unchanged=True,LM_route_unchanged=True,no_additional_packed_weight_payload=True,static_transpose_inside_DeviceMemoryProfiler=True,
        old_dense_binding_kernel_fields_identify_original_NT_module=True,actual_NN_execution_evidence_in_this_route_record=True,Python_calls_do_not_count_Graph_replays=True)
    pointers={}
    def save():
        path=directory/f'route_{os.getpid()}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
    def tensor_sha(t):return hashlib.sha256(memoryview(t.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()
    def profiler():
        frame=inspect.currentframe().f_back
        try:
            while frame:
                if frame.f_globals.get('__name__')=='vllm_fl.worker.model_runner' and frame.f_code.co_name=='load_model' and 'm' in frame.f_locals:
                    p=frame.f_locals['m'];assert hasattr(p,'initial_memory') and not hasattr(p,'final_memory')
                    return dict(source=str(Path(frame.f_code.co_filename).resolve()),sha256=sha(frame.f_code.co_filename),initial_memory=int(p.initial_memory),active=True)
                frame=frame.f_back
            raise AssertionError('No active native model memory profiler')
        finally:del frame
    @functools.wraps(original_process)
    def process(method,layer):
        value=original_process(method,layer)
        if not hasattr(layer,'_tianshu_dense_w8_bank') or tuple(layer.weight.shape)!=(12288,2048):return value
        prefix=layer.prefix;packed=layer._tianshu_dense_w8_packed;version=int(layer.weight._version);assert packed.is_contiguous() and tuple(packed.shape)==(12288,2048)
        proof=profiler();start=time.perf_counter();expected=tensor_sha(packed);size=packed.numel()*packed.element_size()
        if mode=='candidate':
            transposed=packed.T.contiguous();replacement=transposed.T;assert torch.equal(replacement,packed)
            layer._tianshu_dense_w8_packed=replacement
        actual=layer._tianshu_dense_w8_packed;assert tensor_sha(actual)==expected and actual.numel()*actual.element_size()==size and int(layer.weight._version)==version
        assert tuple(actual.stride())==((1,12288) if mode=='candidate' else (2048,1))
        assert actual.data_ptr()%256==0
        item=dict(prefix=prefix,bank_id=int(layer._tianshu_dense_w8_bank),logical_shape=list(actual.shape),logical_stride=list(actual.stride()),logical_sha256=expected,payload_bytes=size,inside_profiler=proof,
            full_device_transpose_equal=(mode=='candidate'),full_logical_hash_equal=True,original_BF16_weight_version_preserved=True,static_layout_seconds=time.perf_counter()-start)
        pointers[actual.data_ptr()]=item;record['banks'].append(item);assert len(record['banks'])<=42
        del packed
        if len(record['banks'])==42:torch.cuda.empty_cache()
        save();return value
    UnquantizedLinearMethod.process_weights_after_loading=process
    @functools.wraps(original_mm)
    def mm(q,w,ascale,wscale,out,integer,config,return_int=False):
        if tuple(w.shape)!=(12288,2048):return original_mm(q,w,ascale,wscale,out,integer,config,return_int)
        bank=pointers[w.data_ptr()];m=q.shape[0];assert m in (8,64)
        if mode=='candidate':
            assert w.stride()==(1,12288) and w.T.is_contiguous();compiled=nn.mm(q,w.T,ascale,wscale,out,integer,configs[m],return_int)
        else:compiled=original_mm(q,w,ascale,wscale,out,integer,config,return_int)
        capturing=torch.cuda.is_current_stream_capturing();module=sys.modules.get('_tianshu_long_segments_actual_graph_trace');graph_key=getattr(module,'STATE',{}).get('active_graph');key=str((bank['bank_id'],m,capturing,graph_key))
        row=record['routes'].setdefault(key,dict(prefix=bank['prefix'],M=m,actually_inside_capture=capturing,graph_key=graph_key,implementation=('NN' if mode=='candidate' else 'NT'),configuration=(configs[m] if mode=='candidate' else config),kernel_source=(str(LAYOUT/'kernel_nn.py') if mode=='candidate' else record['NT_source']),kernel_sha256=(record['NN_sha256'] if mode=='candidate' else record['NT_sha256']),binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared,calls=0))
        row['calls']+=1
        if not capturing and row['calls']&(row['calls']-1)==0:save()
        return compiled
    kernel.mm=mm
    original_capture=ModelRunnerFL.capture_model
    @functools.wraps(original_capture)
    def capture(owner,*args,**kwargs):
        value=original_capture(owner,*args,**kwargs);assert len(record['banks'])==42
        for m in (8,64):assert len({r['prefix'] for r in record['routes'].values() if r['M']==m and r['actually_inside_capture']})==42
        record.update(completed=True,after_original_capture=True,actual_model_memory_usage_bytes=int(owner.model_memory_usage));save();return value
    ModelRunnerFL.capture_model=capture;atexit.register(save);save()
