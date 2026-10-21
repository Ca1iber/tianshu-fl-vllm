# SPDX-License-Identifier: Apache-2.0
"""Default exact Inductor compile sizes, selected before native config deepcopy."""
import functools,hashlib,inspect,json,os
from pathlib import Path
ARG_SHA='f1a32e2207e9ed2f0d954e3e6457effca17ca907cb0cf17bba2ca8bde855a719'
MODEL_SHA='59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180'
MODEL=Path('/workspace/MiniCPM5-2B/config.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eligible(module,owner,usage):
    c=getattr(owner,'compilation_config',None)
    mode=getattr(c,'cudagraph_mode',None)
    return (getattr(module.current_platform,'vendor_name',None)=='iluvatar'
        and usage is not None and usage==getattr(type(usage),'OPENAI_API_SERVER',None)
        and getattr(owner,'model',None)=='/workspace/MiniCPM5-2B' and getattr(owner,'max_model_len',None)==131072
        and str(getattr(owner,'dtype',None)) in ('auto','bfloat16','torch.bfloat16')
        and getattr(owner,'quantization',None) is None and getattr(owner,'speculative_config',None) is None
        and not getattr(owner,'enable_lora',False) and not getattr(owner,'enforce_eager',False)
        and getattr(owner,'tensor_parallel_size',None)==getattr(owner,'pipeline_parallel_size',None)==getattr(owner,'data_parallel_size',None)==1
        and getattr(owner,'performance_mode',None) in (None,'balanced')
        and c is not None and getattr(mode,'name',str(mode))=='FULL_DECODE_ONLY'
        and sha(MODEL)==MODEL_SHA)
def install(module,mode,directory):
    assert mode in ('control','candidate') and sha(module.__file__)==ARG_SHA
    cls=module.EngineArgs;assert not getattr(cls,'_tianshu_static_compile_default',False)
    original=cls.create_engine_config;signature=inspect.signature(original);rows=[];directory=Path(directory)
    @functools.wraps(original)
    def configure(owner,*args,**kwargs):
        bound=signature.bind(owner,*args,**kwargs).arguments
        usage=bound.get('usage_context');c=getattr(owner,'compilation_config',None)
        before=getattr(c,'compile_sizes',None);supported=eligible(module,owner,usage)
        applied=mode=='candidate' and supported and before is None
        if applied:c.compile_sizes=[8,64]
        try:answer=original(owner,*args,**kwargs)
        finally:
            if applied:c.compile_sizes=before
        actual=list(answer.compilation_config.compile_sizes or [])
        if applied:assert sorted(actual)==[8,64]
        explicit_preserved=(before is None or getattr(c,'compile_sizes',None)==before);assert explicit_preserved
        rows.append(dict(mode=mode,eligible=supported,default_applied=applied,before_compile_sizes=before,
            actual_compile_sizes=actual,original_EngineArgs_compile_sizes_restored=(getattr(c,'compile_sizes',None)==before),
            explicit_compile_sizes_preserved=explicit_preserved,actual_budget=answer.scheduler_config.max_num_batched_tokens,
            actual_max_num_seqs=answer.scheduler_config.max_num_seqs,original_arguments_and_return_value_forwarded=True,
            native_source=str(Path(module.__file__).resolve()),native_source_sha256=ARG_SHA,
            policy_source=str(Path(__file__).resolve()),policy_sha256=sha(__file__),configuration_set_before_native_deepcopy=True,
            RNG_and_request_data_not_accessed=True))
        directory.mkdir(parents=True,exist_ok=True);(directory/f'config_{os.getpid()}.json').write_text(json.dumps(rows,indent=2)+'\n')
        return answer
    cls.create_engine_config=configure;cls._tianshu_static_compile_default=True
