# SPDX-License-Identifier: Apache-2.0
"""Configuration-boundary checks only; no throughput prediction."""
import ast
from enum import Enum
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import policy
SOURCE=Path('/usr/local/lib/python3.12/site-packages/vllm/engine/arg_utils.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==policy.budget.ARG_UTILS_SHA
cls=next(x for x in ast.parse(SOURCE.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='EngineArgs')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_set_default_max_num_seqs_and_batched_tokens_args')
class Usage(Enum):OPENAI_API_SERVER='openai_api_server';LLM_CLASS='llm_class'
logger=SimpleNamespace(debug=lambda *a,**k:None,info=lambda *a,**k:None)
ns=dict(logger=logger,SchedulerConfig=SimpleNamespace(DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP=2048,DEFAULT_MAX_NUM_BATCHED_TOKENS=2048,DEFAULT_MAX_NUM_SEQS=256))
future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future,method],type_ignores=[])),str(SOURCE),'exec'),ns)
native=ns[method.name]

def make(directory,mode='candidate',vendor='iluvatar',watermark=0.0,budget=None):
    class Args:
        _set_default_max_num_seqs_and_batched_tokens_args=native
        @classmethod
        def get_batch_defaults(cls,world_size):
            return {Usage.OPENAI_API_SERVER:2048,Usage.LLM_CLASS:8192},{Usage.OPENAI_API_SERVER:256,Usage.LLM_CLASS:256}
    owner=Args()
    owner.__dict__.update(watermark=watermark,max_num_batched_tokens=budget,max_num_seqs=None,performance_mode='balanced',enable_chunked_prefill=True,pipeline_parallel_size=1,tensor_parallel_size=1,speculative_config=None,enable_lora=False)
    model=SimpleNamespace(model='/workspace/MiniCPM5-2B',max_model_len=131072,dtype='torch.bfloat16',quantization=None,is_multimodal_model=False,is_mm_prefix_lm=False,hf_config=SimpleNamespace(hidden_size=2048,num_hidden_layers=42,num_attention_heads=16,num_key_value_heads=2))
    parallel=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,use_batched_dp_moe=False)
    module=SimpleNamespace(__file__=str(SOURCE),EngineArgs=Args,current_platform=SimpleNamespace(vendor_name=vendor))
    policy.budget.install(module,'candidate',directory/'budget')
    policy.install(module,mode,directory/'headroom')
    return owner,model,parallel

checks=[]
with tempfile.TemporaryDirectory(dir='/workspace/benchmark-results',prefix='headroom_cpu_') as temp:
    for name,kwargs,change,expected in (
        ('selected',{},None,policy.RATIO),
        ('control',dict(mode='control'),None,0.0),
        ('existing_positive',dict(watermark=.125),None,.125),
        ('other_vendor',dict(vendor='metax'),None,0.0),
        ('other_model',{},lambda m,p:setattr(m,'model','/other'),0.0),
        ('parallel2',{},lambda m,p:setattr(p,'tensor_parallel_size',2),0.0),
        ('explicit4096',dict(budget=4096),None,policy.RATIO)):
        owner,model,parallel=make(Path(temp)/name,**kwargs)
        if change:change(model,parallel)
        answer=owner._set_default_max_num_seqs_and_batched_tokens_args(usage_context=Usage.OPENAI_API_SERVER,model_config=model,parallel_config=parallel)
        assert answer is None and owner.watermark==expected,name
        if name in ('selected','control','explicit4096'):assert owner.max_num_batched_tokens==4096
        row=json.loads(next((Path(temp)/name/'headroom').glob('*.json')).read_text())[-1]
        assert row['applied']==(expected==policy.RATIO)
        checks.append(name)
    owner,model,parallel=make(Path(temp)/'rollback')
    sentinel=RuntimeError('native failure')
    def fail(*a,**k):raise sentinel
    previous=logger.debug
    logger.debug=fail
    try:
        try:owner._set_default_max_num_seqs_and_batched_tokens_args(Usage.OPENAI_API_SERVER,model,parallel)
        except RuntimeError as observed:assert observed is sentinel
        else:raise AssertionError('exception swallowed')
        assert owner.watermark==0.0 and owner.max_num_batched_tokens is None
        checks.append('exception_identity_and_two_defaults_rollback')
    finally:logger.debug=previous
base=Path(__file__).resolve().parent
for path in base.rglob('*.py'):ast.parse(path.read_text())
sources=[*base.rglob('*.py'),SOURCE,Path(policy.budget.__file__),Path('/usr/local/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py')]
result=dict(complete=True,CPU_only=True,GPU_executed=False,adopted=False,guard_cases=checks,
            native_default_setting_AST_executed=True,budget4096_composition_verified=True,
            native_KV_manager_not_modified=True,candidate_watermark=policy.RATIO,
            capacity_tokens_observed_from_unchanged_16k=678176,
            illustrative_headroom_tokens=678176*policy.RATIO,
            illustrative_uniform16k_prefilled_capacity_without_decoding=int(678176*(1-policy.RATIO)//16384),
            illustration_is_not_scheduler_simulation_or_service_result=True,
            frozen_sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
(base/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='frozen_sources'}))
