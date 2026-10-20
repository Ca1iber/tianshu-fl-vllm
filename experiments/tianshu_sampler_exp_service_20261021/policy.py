# SPDX-License-Identifier: Apache-2.0
"""Call unchanged capture once, then optionally warm the native sampling cache."""
import functools,hashlib,importlib.util,inspect,json,os
from pathlib import Path
BASE=Path(__file__).resolve().parent
COMPONENT=BASE.parent/'tianshu_sampler_exp_prewarm_20261021'

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
    assert cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==cfg.parallel_config.data_parallel_size==1
    assert cfg.speculative_config is None and cfg.lora_config is None
    assert cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.async_scheduling is True
    assert os.environ['TIANSHU_LM_PRECISION_MODE']=='candidate' and os.environ['VLLM_DISABLE_COMPILE_CACHE']=='1'
    gate=json.loads((COMPONENT/'GPU_GATE_32g.json').read_text());assert gate['complete'] and gate['default_CPU_and_GPU_RNG_exact'] and gate['actual_B37_FP32_words']==4830720
    assert sha(COMPONENT/'warmup.py')==gate['source_snapshots'][str(COMPONENT/'warmup.py')]
    spec=importlib.util.spec_from_file_location('_sampler_warm_startup_primitive',COMPONENT/'warmup.py')
    warmup=importlib.util.module_from_spec(spec);spec.loader.exec_module(warmup)
    assert not getattr(ModelRunnerFL,'_tianshu_sampler_exp_warm_installed',False)
    original=ModelRunnerFL.capture_model;native=Path(inspect.getsourcefile(original)).resolve()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    record=dict(mode=mode,installed=True,phase='installed',completed=False,activation_count=0,capture_calls=0,
        original_capture_source=str(native),original_capture_sha256=sha(native),policy_source=str(Path(__file__).resolve()),policy_sha256=sha(__file__),
        helper_source=str(COMPONENT/'warmup.py'),helper_sha256=sha(COMPONENT/'warmup.py'),actual_budget=4096,actual_max_seqs=64,
        model_precision_mode='candidate',RNG_algorithm_not_modified=True,request_data_not_accessed=True)
    def save():
        target=directory/f'route_{os.getpid()}.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(target)
    save()
    @functools.wraps(original)
    def capture(owner,*args,**kwargs):
        answer=original(owner,*args,**kwargs);record['capture_calls']+=1
        if record['completed']:save();return answer
        assert not torch.cuda.is_current_stream_capturing()
        import flag_gems
        operator=flag_gems.exponential_;path=Path(inspect.getsourcefile(operator)).resolve()
        assert path==Path('/usr/local/lib/python3.12/site-packages/flag_gems/ops/exponential_.py')
        assert sha(path)==gate['source_snapshots'][str(path)]
        record.update(phase='warming',native_operator_source=str(path),native_operator_sha256=sha(path),after_original_capture_returned=True);save()
        cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
        if mode=='candidate':record['prewarm']=warmup.prewarm(torch,operator)
        else:record['prewarm']=dict(complete=True,skipped_in_control=True,calls=[],elapsed_s=0.)
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
        record.update(phase='complete',completed=True,activation_count=1,CPU_RNG_before=warmup.tensor_sha(cpu),CPU_RNG_after=warmup.tensor_sha(torch.get_rng_state()),
            GPU_RNG_before=warmup.tensor_sha(gpu),GPU_RNG_after=warmup.tensor_sha(torch.cuda.get_rng_state()),default_RNG_unchanged=True,original_capture_return_preserved=True)
        save();return answer
    ModelRunnerFL.capture_model=capture;ModelRunnerFL._tianshu_sampler_exp_warm_installed=True
