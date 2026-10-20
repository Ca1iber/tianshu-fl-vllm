# SPDX-License-Identifier: Apache-2.0
"""Guarded default chunk cap plus identical scheduling observation in both arms."""
import functools,hashlib,importlib.util,inspect,json,os
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_chunk_original_default_guard',BASE.parent/'tianshu_k8v16_budget_default_20261019/default_policy.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
SCHED_SHA='df78076164fe02a63876d7690cefac810f1ba4d8fb58afc15e786c1608064090'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def install_defaults(module,mode,directory):
    assert mode in ('control','candidate') and sha(module.__file__)==parent.ARG_UTILS_SHA
    cls=module.EngineArgs;assert not getattr(cls,'_tianshu_prefill_chunk_default_installed',False)
    original=cls._set_default_max_num_seqs_and_batched_tokens_args;signature=inspect.signature(original);rows=[]
    @functools.wraps(original)
    def select(owner,*args,**kwargs):
        bound=signature.bind(owner,*args,**kwargs).arguments
        usage,model,parallel=(bound[n] for n in ('usage_context','model_config','parallel_config'))
        before=owner.long_prefill_token_threshold;before_budget=owner.max_num_batched_tokens
        supported=parent.eligible(module,owner,usage,model,parallel)
        applied=supported and before==0 and before_budget is None
        answer=original(owner,*args,**kwargs)
        if applied:
            assert owner.max_num_batched_tokens==4096 and owner.max_num_seqs==64
            owner.long_prefill_token_threshold=0 if mode=='control' else 1024
        rows.append(dict(mode=mode,eligible=supported,default_selected=applied,before_threshold=before,after_threshold=owner.long_prefill_token_threshold,
            before_budget=before_budget,actual_budget=owner.max_num_batched_tokens,actual_max_seqs=owner.max_num_seqs,
            nonzero_explicit_threshold_preserved=before==0 or before==owner.long_prefill_token_threshold,
            original_called_once=True,original_arguments_and_return_forwarded=True,
            native_source=str(Path(module.__file__).resolve()),native_source_sha256=parent.ARG_UTILS_SHA,policy_source=str(Path(__file__).resolve()),policy_sha256=sha(__file__)))
        save(Path(directory)/f'default_{os.getpid()}.json',rows);return answer
    cls._set_default_max_num_seqs_and_batched_tokens_args=select;cls._tianshu_prefill_chunk_default_installed=True

def install_scheduler(module,mode,directory):
    assert mode in ('control','candidate') and sha(module.__file__)==SCHED_SHA
    cls=module.Scheduler;assert not getattr(cls,'_tianshu_prefill_chunk_observer_installed',False)
    original_init=cls.__init__
    @functools.wraps(original_init)
    def init(owner,*args,**kwargs):
        answer=original_init(owner,*args,**kwargs);cfg=owner.vllm_config;model=cfg.model_config;hf=model.hf_config
        supported=model.model=='/workspace/MiniCPM5-2B' and str(model.dtype)=='torch.bfloat16' and model.quantization is None and not model.is_multimodal_model
        supported=supported and (hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads)==(2048,42,16,2)
        supported=supported and cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==cfg.parallel_config.data_parallel_size==1 and cfg.speculative_config is None
        if not supported:return answer
        threshold=0 if mode=='control' else 1024
        assert cfg.scheduler_config.long_prefill_token_threshold==threshold and owner.max_num_scheduled_tokens==4096 and cfg.scheduler_config.max_num_seqs==64
        original_schedule=owner.schedule;native=Path(inspect.getsourcefile(original_schedule)).resolve()
        record=dict(mode=mode,installed=True,actual_threshold=threshold,actual_budget=4096,actual_max_seqs=64,
            scheduler_module_source=str(Path(module.__file__).resolve()),scheduler_module_sha256=SCHED_SHA,
            actual_schedule_source=str(native),actual_schedule_sha256=sha(native),policy_source=str(Path(__file__).resolve()),policy_sha256=sha(__file__),
            steps=0,observed_prefill_events=0,max_observed_prefill_work=0,prefill_histogram={},snapshot_scope='saved on new shape and power-of-two steps',
            owner_progress_read_after_native_update=True,original_schedule_arguments_and_return_forwarded=True)
        path=Path(directory)/f'schedule_{os.getpid()}.json';save(path,record)
        @functools.wraps(original_schedule)
        def observe(*sched_args,**sched_kwargs):
            result=original_schedule(*sched_args,**sched_kwargs);record['steps']+=1;changed=False
            for request_id,count in result.num_scheduled_tokens.items():
                request=owner.requests[request_id];after=request.num_computed_tokens;before=after-count;prompt=request.num_prompt_tokens
                assert before>=0
                if before<prompt:
                    assert count>0
                    if threshold:assert count<=threshold
                    key=str(count);changed=changed or key not in record['prefill_histogram']
                    row=record['prefill_histogram'].setdefault(key,dict(events=0,min_computed_before=before,max_computed_after=after))
                    row['events']+=1;row['min_computed_before']=min(row['min_computed_before'],before);row['max_computed_after']=max(row['max_computed_after'],after)
                    record['observed_prefill_events']+=1;record['max_observed_prefill_work']=max(record['max_observed_prefill_work'],count)
            step=record['steps']
            if changed or step&(step-1)==0:save(path,record)
            return result
        owner.schedule=observe
        return answer
    cls.__init__=init;cls._tianshu_prefill_chunk_observer_installed=True
