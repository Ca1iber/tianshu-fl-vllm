# SPDX-License-Identifier: Apache-2.0
"""Run installed EngineArgs default AST with actual inherited guards, without GPU."""
import ast,hashlib,importlib.util,json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from candidate_profile import BASE,prepare
from policy import install,parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    source=Path('/usr/local/lib/python3.12/site-packages/vllm/engine/arg_utils.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==parent.ARG_UTILS_SHA
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EngineArgs')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_set_default_max_num_seqs_and_batched_tokens_args')
    class Usage(Enum):OPENAI_API_SERVER='openai_api_server';LLM_CLASS='llm_class'
    logger=SimpleNamespace(debug=lambda *a,**k:None,info=lambda *a,**k:None)
    namespace=dict(logger=logger,SchedulerConfig=SimpleNamespace(DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP=2048,DEFAULT_MAX_NUM_BATCHED_TOKENS=2048,DEFAULT_MAX_NUM_SEQS=256))
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,method],type_ignores=[])),str(source),'exec'),namespace)
    native=namespace[method.name]
    seq=load('_latest_budget_CPU_actual_seq_guard',BASE.parent/'tianshu_seq_limit_default_20261020/policy.py')
    root=Path('/workspace/benchmark-results/20261021_dense_budget8192_CPU_guard_v2_tianshu');assert not root.exists();root.mkdir()
    def make(name,mode='candidate',vendor='iluvatar',budget=None,seqs=None,performance='balanced',chunked=True,wrappers=True):
        class Args:
            _set_default_max_num_seqs_and_batched_tokens_args=native
            @classmethod
            def get_batch_defaults(cls,world_size):
                return {Usage.OPENAI_API_SERVER:2048,Usage.LLM_CLASS:8192},{Usage.OPENAI_API_SERVER:256,Usage.LLM_CLASS:256}
        owner=Args();owner.__dict__.update(max_num_batched_tokens=budget,max_num_seqs=seqs,performance_mode=performance,enable_chunked_prefill=chunked,pipeline_parallel_size=1,tensor_parallel_size=1,speculative_config=None,enable_lora=False)
        model=SimpleNamespace(model='/workspace/MiniCPM5-2B',max_model_len=131072,dtype='torch.bfloat16',quantization=None,is_multimodal_model=False,is_mm_prefix_lm=False,hf_config=SimpleNamespace(hidden_size=2048,num_hidden_layers=42,num_attention_heads=16,num_key_value_heads=2))
        parallel=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,use_batched_dp_moe=False)
        module=SimpleNamespace(__file__=str(source),EngineArgs=Args,current_platform=SimpleNamespace(vendor_name=vendor))
        directory=root/name
        if wrappers:
            parent.install(module,'candidate',directory/'inner_budget');seq.install(module,'candidate',directory/'inner_seq');install(module,mode,directory/'outer_budget')
        return owner,model,parallel,directory
    checks=[]
    for name,kwargs,change,expected,expected_seqs,applied in (
        ('control',dict(mode='control'),None,4096,64,True),('candidate',{},None,8192,64,True),
        ('explicit2048',dict(budget=2048),None,2048,64,False),('explicit4096',dict(budget=4096),None,4096,64,False),('explicit8192',dict(budget=8192),None,8192,64,False),
        ('other_vendor',dict(vendor='metax'),None,2048,256,False),('other_model',{},lambda m,p:setattr(m,'model','/other'),2048,256,False),
        ('other_geometry',{},lambda m,p:setattr(m.hf_config,'hidden_size',4096),2048,256,False),('FP16',{},lambda m,p:setattr(m,'dtype','torch.float16'),2048,256,False),
        ('parallel2',{},lambda m,p:setattr(p,'tensor_parallel_size',2),2048,256,False),('throughput',dict(performance='throughput'),None,4096,512,False),
        ('chunked_off',dict(chunked=False),None,131072,256,False),('explicit512seqs',dict(seqs=512),None,2048,512,False)):
        owner,model,parallel,directory=make(name,**kwargs)
        if change:change(model,parallel)
        answer=owner._set_default_max_num_seqs_and_batched_tokens_args(usage_context=Usage.OPENAI_API_SERVER,model_config=model,parallel_config=parallel)
        assert answer is None and owner.max_num_batched_tokens==expected and owner.max_num_seqs==expected_seqs,name
        row=json.loads(next((directory/'outer_budget').glob('*.json')).read_text())[-1]
        assert row['default_applied']==applied and row['after_budget']==expected and row['explicit_budget_preserved'],name
        checks.append(name)
    for mode,budget in (('control',4096),('candidate',8192)):
        selected,model,parallel,_=make('state_'+mode,mode=mode)
        selected._set_default_max_num_seqs_and_batched_tokens_args(Usage.OPENAI_API_SERVER,model,parallel)
        explicit,model,parallel,_=make('explicit_state_'+mode,budget=budget,seqs=64,wrappers=False)
        explicit._set_default_max_num_seqs_and_batched_tokens_args(Usage.OPENAI_API_SERVER,model,parallel)
        assert selected.__dict__==explicit.__dict__,mode
        failing,model,parallel,_=make('exception_'+mode,mode=mode);error=RuntimeError('native default failure')
        def fail(*args,**kwargs):raise error
        previous=logger.debug;logger.debug=fail
        try:
            try:failing._set_default_max_num_seqs_and_batched_tokens_args(Usage.OPENAI_API_SERVER,model,parallel)
            except RuntimeError as observed:assert observed is error
            else:raise AssertionError('exception swallowed')
            assert failing.max_num_batched_tokens is None
        finally:logger.debug=previous
    profiles=[]
    for mode in ('control','candidate'):
        target=Path(f'/workspace/benchmark-results/20261021_dense_budget8192_CPU_profile_{mode}_tianshu');assert not target.exists();target.mkdir()
        command,env,profile,sources=prepare(target,mode)
        assert len(profile)==64 and len(command)==13 and profile['TIANSHU_DENSE_BUDGET_MODE']==mode
        assert all(profile[k]=='candidate' for k in ('TIANSHU_DENSE_INT8_MODE','TIANSHU_LM_PRECISION_MODE','TIANSHU_PREFILL_LINEAR_MODE','TIANSHU_SEQ_LIMIT_MODE'))
        profiles.append(dict(mode=mode,command=command,profile_environment=profile,frozen_sources=sources))
    assert profiles[0]['frozen_sources']==profiles[1]['frozen_sources']
    for path in BASE.rglob('*.py'):ast.parse(path.read_text())
    result=dict(complete=True,CPU_only=True,no_GPU_started=True,guard_cases=checks,
        actual_installed_EngineArgs_AST_executed=True,actual_inherited_budget_and_seq_wrappers_executed=True,
        selected_state_equals_explicit8192_seq64=True,control_state_equals_explicit4096_seq64=True,
        explicit_budget_preserved=True,native_exception_identity_and_rollback_verified=True,
        parent_qualified_runtime_sources=359,runtime_source_count=len(profiles[0]['frozen_sources']),environment_count=64,
        fresh_Math_required=True,profiles=profiles)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
if __name__=='__main__':main()
