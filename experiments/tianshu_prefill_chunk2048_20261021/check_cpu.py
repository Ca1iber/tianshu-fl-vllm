# SPDX-License-Identifier: Apache-2.0
"""Execute installed EngineArgs AST and native wrapper forwarding on CPU."""
import ast,importlib.util,json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from candidate_profile import BASE,prepare,digest,read
from policy import parent,install_defaults,install_scheduler,SCHED_SHA

def load(n,p):
    spec=importlib.util.spec_from_file_location(n,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    for path in BASE.rglob('*.py'):ast.parse(path.read_text())
    source=Path('/usr/local/lib/python3.12/site-packages/vllm/engine/arg_utils.py');assert digest(source)==parent.ARG_UTILS_SHA
    scheduler_path=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py');assert digest(scheduler_path)==SCHED_SHA
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EngineArgs');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_set_default_max_num_seqs_and_batched_tokens_args')
    class Usage(Enum):OPENAI_API_SERVER='openai_api_server';LLM_CLASS='llm_class'
    logger=SimpleNamespace(debug=lambda *a,**k:None,info=lambda *a,**k:None)
    ns=dict(logger=logger,SchedulerConfig=SimpleNamespace(DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP=2048,DEFAULT_MAX_NUM_BATCHED_TOKENS=2048,DEFAULT_MAX_NUM_SEQS=256))
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,method],type_ignores=[])),str(source),'exec'),ns);native=ns[method.name]
    seq=load('_chunk_CPU_actual_seq',BASE.parent/'tianshu_seq_limit_default_20261020/policy.py')
    root=Path('/workspace/benchmark-results/20261021_prefill_chunk2048_CPU_guard_v2_tianshu');assert not root.exists();root.mkdir()
    def make(name,mode='candidate',vendor='iluvatar',budget=None,seqs=None,threshold=0,chunked=True):
        class Args:
            _set_default_max_num_seqs_and_batched_tokens_args=native
            @classmethod
            def get_batch_defaults(cls,world_size):return {Usage.OPENAI_API_SERVER:2048,Usage.LLM_CLASS:8192},{Usage.OPENAI_API_SERVER:256,Usage.LLM_CLASS:256}
        owner=Args();owner.__dict__.update(max_num_batched_tokens=budget,max_num_seqs=seqs,long_prefill_token_threshold=threshold,performance_mode='balanced',enable_chunked_prefill=chunked,pipeline_parallel_size=1,tensor_parallel_size=1,speculative_config=None,enable_lora=False)
        model=SimpleNamespace(model='/workspace/MiniCPM5-2B',max_model_len=131072,dtype='torch.bfloat16',quantization=None,is_multimodal_model=False,is_mm_prefix_lm=False,hf_config=SimpleNamespace(hidden_size=2048,num_hidden_layers=42,num_attention_heads=16,num_key_value_heads=2))
        parallel=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,use_batched_dp_moe=False)
        module=SimpleNamespace(__file__=str(source),EngineArgs=Args,current_platform=SimpleNamespace(vendor_name=vendor))
        parent.install(module,'candidate',root/name/'budget');seq.install(module,'candidate',root/name/'seq');install_defaults(module,mode,root/name/'chunk')
        return owner,model,parallel
    cases=[]
    for name,kwargs,change,expected in (
        ('control',dict(mode='control'),None,0),('candidate',{},None,2048),('explicit512',dict(threshold=512),None,512),('explicit2048',dict(threshold=2048),None,2048),
        ('budget2048',dict(budget=2048),None,0),('seq64',dict(seqs=64),None,0),('metax',dict(vendor='metax'),None,0),
        ('othermodel',{},lambda m,p:setattr(m,'model','/other'),0),('parallel2',{},lambda m,p:setattr(p,'tensor_parallel_size',2),0),('chunkoff',dict(chunked=False),None,0)):
        owner,model,parallel=make(name,**kwargs)
        if change:change(model,parallel)
        assert owner._set_default_max_num_seqs_and_batched_tokens_args(Usage.OPENAI_API_SERVER,model,parallel) is None and owner.long_prefill_token_threshold==expected,name
        if name in ('control','candidate'):assert owner.max_num_batched_tokens==4096 and owner.max_num_seqs==64
        cases.append(name)
    owner,model,parallel=make('exception');error=RuntimeError('original default error');previous=logger.debug
    def fail(*a,**k):raise error
    logger.debug=fail
    try:
        try:owner._set_default_max_num_seqs_and_batched_tokens_args(Usage.OPENAI_API_SERVER,model,parallel)
        except RuntimeError as observed:assert observed is error and owner.long_prefill_token_threshold==0
        else:raise AssertionError('exception swallowed')
    finally:logger.debug=previous
    for mode,threshold in (('control',0),('candidate',2048)):
        class Scheduler:
            def __init__(self,cfg):
                self.vllm_config=cfg;self.max_num_scheduled_tokens=4096;self.requests={'r':SimpleNamespace(num_computed_tokens=0,num_prompt_tokens=4096)};self.value=1024
            def schedule(self):
                self.requests['r'].num_computed_tokens+=self.value
                return SimpleNamespace(num_scheduled_tokens={'r':self.value})
        _,model,parallel=make('fake_models_'+mode,mode=mode)
        cfg=SimpleNamespace(model_config=model,parallel_config=parallel,speculative_config=None,scheduler_config=SimpleNamespace(long_prefill_token_threshold=threshold,max_num_seqs=64))
        module=SimpleNamespace(__file__=str(scheduler_path),Scheduler=Scheduler);install_scheduler(module,mode,root/('observer_'+mode));scheduler=Scheduler(cfg)
        result=scheduler.schedule();assert result.num_scheduled_tokens=={'r':1024}
        record=read(next((root/('observer_'+mode)).glob('schedule_*.json')));assert record['observed_prefill_events']==1 and record['max_observed_prefill_work']==1024
        if threshold:
            scheduler.value=4096
            try:scheduler.schedule()
            except AssertionError:pass
            else:raise AssertionError('observer failed to reject excess chunk')
    profiles=[]
    for mode in ('control','candidate'):
        root=Path(f'/workspace/benchmark-results/20261021_prefill_chunk2048_CPU_profile_{mode}_tianshu');assert not root.exists();root.mkdir()
        command,environment,profile,sources=prepare(root,mode);assert len(command)==13 and len(profile)==62
        profiles.append(dict(mode=mode,root=str(root),command=command,environment=profile,sources=sources))
    assert profiles[0]['sources']==profiles[1]['sources']
    a,b=profiles;expected={k:v.replace(a['root']+'/',b['root']+'/').replace('/hooks_control','/hooks_candidate') for k,v in a['environment'].items()};assert expected==b['environment']
    gate=dict(complete=True,CPU_only=True,no_GPU_started=True,adopted=False,new_Math_not_run=True,actual_installed_EngineArgs_AST_executed=True,
        inherited_budget_and_seq_wrappers_used=True,nonzero_explicit_threshold_preserved=True,original_exception_identity_preserved=True,
        actual_observer_code_executed_on_synthetic_scheduler=True,observer_rejects_excess_candidate_chunk=True,guard_cases=cases,
        runtime_source_count=len(b['sources']),environment_count=62,profiles=profiles)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='profiles'}))
if __name__=='__main__':main()
