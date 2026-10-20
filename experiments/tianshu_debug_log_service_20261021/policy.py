# SPDX-License-Identifier: Apache-2.0
"""After unmodified capture, retain all handlers and suppress only FlagGems DEBUG."""
import functools,hashlib,importlib,importlib.util,inspect,json,logging,os
from pathlib import Path
BASE=Path(__file__).resolve().parent
COMPONENT=BASE.parent/'tianshu_debug_log_level_20261021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install(worker,mode,directory):
    assert mode in ('control','candidate')
    from vllm.platforms import current_platform
    from vllm_fl.worker.model_runner import ModelRunnerFL
    cfg=worker.vllm_config;model=cfg.model_config;hf=model.hf_config
    assert current_platform.vendor_name=='iluvatar' and model.model=='/workspace/MiniCPM5-2B'
    assert (hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads,hf.vocab_size)==(2048,42,16,2,130560)
    assert str(model.dtype)=='torch.bfloat16' and model.quantization is None
    assert cfg.speculative_config is None and cfg.lora_config is None
    assert cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==cfg.parallel_config.data_parallel_size==1
    assert cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.async_scheduling
    assert os.environ['TIANSHU_LM_PRECISION_MODE']=='candidate' and os.environ['VLLM_DISABLE_COMPILE_CACHE']=='1'
    gate=json.loads((COMPONENT/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['all_INFO_WARNING_ERROR_CRITICAL_records_and_text_preserved']
    for path,digest in gate['sources'].items():assert sha(path)==digest
    spec=importlib.util.spec_from_file_location('_debug_log_actual_threshold_helper',COMPONENT/'level.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    cls=ModelRunnerFL;assert not getattr(cls,'_tianshu_debug_log_level_installed',False)
    original_capture,original_sample=cls.capture_model,cls.sample_tokens
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    record=dict(mode=mode,installed=True,completed=False,capture_calls=0,sample_calls=0,
        policy_source=str(Path(__file__).resolve()),policy_sha256=sha(__file__),helper_source=str(COMPONENT/'level.py'),helper_sha256=sha(COMPONENT/'level.py'),
        original_capture_source=str(Path(inspect.getsourcefile(original_capture)).resolve()),original_capture_sha256=sha(inspect.getsourcefile(original_capture)),
        original_sample_source=str(Path(inspect.getsourcefile(original_sample)).resolve()),original_sample_sha256=sha(inspect.getsourcefile(original_sample)),
        actual_budget=4096,actual_max_seqs=64,model_precision_mode='candidate',RNG_and_request_data_not_accessed=True)
    logger=logging.getLogger('flag_gems');children=[logging.getLogger('flag_gems.ops.'+name) for name in ('add','alias','copy','fill','index','sub','to')]
    def save():
        path=directory/f'route_{os.getpid()}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
    save()
    @functools.wraps(original_capture)
    def capture(owner,*args,**kwargs):
        answer=original_capture(owner,*args,**kwargs);record['capture_calls']+=1
        if record['completed']:save();return answer
        assert logger.getEffectiveLevel()==logging.DEBUG and all(c.level==logging.NOTSET for c in children)
        assert any(getattr(h,'_flaggems_owned',False) and any(type(f).__name__=='LogOncePerLocationFilter' for f in h.filters) for h in logger.handlers)
        from flag_gems import runtime
        assert 'index_adjacent' not in runtime.config_loader.loaded_triton_config
        record['threshold_change']=helper.apply(logger,mode)
        record.update(completed=True,after_original_capture_returned=True,original_capture_return_preserved=True,
            INFO_and_above_enabled=True,index_configuration_cache_unchanged_missing=True,
            native_logger_source=str(Path(importlib.import_module('flag_gems.logging_utils').__file__).resolve()))
        record['native_logger_sha256']=sha(record['native_logger_source']);save();return answer
    @functools.wraps(original_sample)
    def sample(owner,*args,**kwargs):
        answer=original_sample(owner,*args,**kwargs)
        if record['completed']:
            record['sample_calls']+=1
            expected=logging.DEBUG if mode=='control' else logging.INFO
            assert logger.getEffectiveLevel()==expected and all(c.getEffectiveLevel()==expected for c in children)
            if (record['sample_calls'] & (record['sample_calls']-1))==0:
                current=helper.describe(logger);initial=record['threshold_change']['after']
                assert current==initial
                assert all(logger.isEnabledFor(level) for level in (logging.INFO,logging.WARNING,logging.ERROR,logging.CRITICAL))
                record.update(actual_logger=current,children=[dict(name=c.name,level=c.level,effective_level=c.getEffectiveLevel()) for c in children],
                    active_level=expected,all_original_sample_arguments_and_results_forwarded=True)
                save()
        return answer
    cls.capture_model=capture;cls.sample_tokens=sample;cls._tianshu_debug_log_level_installed=True
