# SPDX-License-Identifier: Apache-2.0
"""A guarded default at the native configuration boundary; explicit user values win."""
import functools,hashlib,inspect,json,os
from pathlib import Path

ARG_UTILS_SHA='f1a32e2207e9ed2f0d954e3e6457effca17ca907cb0cf17bba2ca8bde855a719'


def eligible(module,owner,usage,model,parallel):
    hf=model.hf_config
    return (getattr(module.current_platform,'vendor_name',None)=='iluvatar'
        and usage is not None and usage==getattr(type(usage),'OPENAI_API_SERVER',None)
        and model.model=='/workspace/MiniCPM5-2B' and model.max_model_len==131072
        and str(model.dtype)=='torch.bfloat16' and model.quantization is None
        and (hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads)==(2048,42,16,2)
        and not model.is_multimodal_model
        and parallel.tensor_parallel_size==parallel.pipeline_parallel_size==parallel.data_parallel_size==1
        and not parallel.use_batched_dp_moe
        and owner.max_num_seqs in (None,256) and owner.enable_chunked_prefill is True
        and owner.performance_mode in ('balanced',None)
        and getattr(owner,'speculative_config',None) is None and not getattr(owner,'enable_lora',False))


def install(module,mode,directory):
    assert mode in ('control','candidate')
    source=Path(module.__file__);assert hashlib.sha256(source.read_bytes()).hexdigest()==ARG_UTILS_SHA
    cls=module.EngineArgs
    assert not getattr(cls,'_tianshu_budget_default_installed',False)
    original=cls._set_default_max_num_seqs_and_batched_tokens_args
    signature=inspect.signature(original)
    records=[]
    @functools.wraps(original)
    def select(owner,*args,**kwargs):
        bound=signature.bind(owner,*args,**kwargs).arguments
        usage,model,parallel=(bound[n] for n in ('usage_context','model_config','parallel_config'))
        supported=eligible(module,owner,usage,model,parallel)
        before=owner.max_num_batched_tokens
        applied=mode=='candidate' and supported and before is None
        if applied:owner.max_num_batched_tokens=4096
        try:
            answer=original(owner,*args,**kwargs)
        except BaseException:
            if applied:owner.max_num_batched_tokens=before
            raise
        if applied:assert owner.max_num_batched_tokens==4096
        records.append(dict(mode=mode,eligible=supported,default_applied=applied,
            before_budget=before,after_budget=owner.max_num_batched_tokens,
            max_num_seqs=owner.max_num_seqs,usage=getattr(usage,'value',None),
            source_path=str(source),source_sha256=ARG_UTILS_SHA,
            original_arguments_forwarded=True,RNG_not_read_or_modified=True))
        directory.mkdir(parents=True,exist_ok=True)
        (directory/f'default_{os.getpid()}.json').write_text(json.dumps(records,indent=2)+'\n')
        return answer
    cls._set_default_max_num_seqs_and_batched_tokens_args=select
    cls._tianshu_budget_default_installed=True
