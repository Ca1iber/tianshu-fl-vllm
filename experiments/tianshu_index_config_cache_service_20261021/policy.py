# SPDX-License-Identifier: Apache-2.0
"""After native capture, cache the exact index fallback config list only in candidate."""
import functools,hashlib,importlib,importlib.util,inspect,json,os
from pathlib import Path
BASE=Path(__file__).resolve().parent
COMPONENT=BASE.parent/'tianshu_index_config_cache_20261021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install(worker,mode,directory):
    assert mode in ('control','candidate')
    import torch
    from vllm.platforms import current_platform
    from vllm_fl.worker.model_runner import ModelRunnerFL
    cfg=worker.vllm_config;model=cfg.model_config;hf=model.hf_config
    assert current_platform.vendor_name=='iluvatar' and model.model=='/workspace/MiniCPM5-2B'
    assert (hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads,hf.vocab_size)==(2048,42,16,2,130560)
    assert str(model.dtype)=='torch.bfloat16' and model.quantization is None and not model.is_multimodal_model
    assert cfg.speculative_config is None and cfg.lora_config is None
    assert cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==cfg.parallel_config.data_parallel_size==1
    assert cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.async_scheduling
    assert os.environ['TIANSHU_LM_PRECISION_MODE']=='candidate' and os.environ['VLLM_DISABLE_COMPILE_CACHE']=='1'
    gate=json.loads((COMPONENT/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['actual_selection_and_generated_source_cases']==16 and gate['all_selected_configs_and_generated_sources_identical']
    for path,digest in gate['sources'].items():assert sha(path)==digest
    spec=importlib.util.spec_from_file_location('_actual_index_config_cache_helper',COMPONENT/'cache.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    cls=ModelRunnerFL;assert not getattr(cls,'_tianshu_index_config_cache_installed',False)
    original=cls.capture_model;native=Path(inspect.getsourcefile(original)).resolve()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    record=dict(mode=mode,installed=True,completed=False,capture_calls=0,lookup_calls=0,construction_misses=0,
        policy_source=str(Path(__file__).resolve()),policy_sha256=sha(__file__),helper_source=str(COMPONENT/'cache.py'),helper_sha256=sha(COMPONENT/'cache.py'),
        original_capture_source=str(native),original_capture_sha256=sha(native),actual_budget=4096,actual_max_seqs=64,
        model_precision_mode='candidate',RNG_and_request_data_not_accessed=True,native_config_values_not_modified=True)
    def save():
        path=directory/f'route_{os.getpid()}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
    save()
    @functools.wraps(original)
    def capture(owner,*args,**kwargs):
        result=original(owner,*args,**kwargs);record['capture_calls']+=1
        if record['completed']:save();return result
        assert not torch.cuda.is_current_stream_capturing()
        from flag_gems import runtime
        loader=runtime.config_loader;op=helper.OP;cache=loader.loaded_triton_config
        original_get=loader.get_tuned_config;before=dict(cache)
        values=original_get(op);expected=helper.signature(values)
        assert expected==gate['full_config_signature'] and op not in cache
        if mode=='candidate':record['prime']=helper.prime(loader)
        record.update(after_original_capture_returned=True,configuration_count=len(values),configuration_signature=expected,
            cache_entry_initially_missing=True,native_config_loader_source=str(Path(inspect.getsourcefile(type(loader))).resolve()),
            native_config_loader_sha256=sha(inspect.getsourcefile(type(loader))))
        def observe(op_name):
            hit=op_name in cache;answer=original_get(op_name)
            if op_name==op:
                record['lookup_calls']+=1;record['construction_misses']+=int(not hit)
                if (record['lookup_calls'] & (record['lookup_calls']-1))==0:
                    assert helper.signature(answer)==expected
                    assert all(cache.get(k) is v for k,v in before.items())
                    record.update(last_signature_matches_CPU_proof=True,preexisting_cache_entry_identities_preserved=True,
                        actual_entry_present=op in cache,actual_result_is_cached_entry=answer is cache.get(op))
                    save()
            return answer
        loader.get_tuned_config=observe
        record.update(completed=True,after_original_capture_returned=True,original_capture_return_preserved=True)
        save();return result
    cls.capture_model=capture;cls._tianshu_index_config_cache_installed=True
