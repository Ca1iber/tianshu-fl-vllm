# SPDX-License-Identifier: Apache-2.0
"""Execute installed native default-setting AST, verify composition and explicit values."""
import ast,hashlib,json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import policy
BASE=Path(__file__).resolve().parent;SOURCE=Path('/usr/local/lib/python3.12/site-packages/vllm/engine/arg_utils.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==policy.budget.ARG_UTILS_SHA
root=Path('/workspace/benchmark-results/20261020_seq_limit_default_CPU_tianshu');root.mkdir(exist_ok=False)
cls=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='EngineArgs')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_set_default_max_num_seqs_and_batched_tokens_args')
class Usage(Enum):OPENAI_API_SERVER='openai_api_server';LLM_CLASS='llm_class'
logger=SimpleNamespace(debug=lambda *a,**k:None,info=lambda *a,**k:None)
ns=dict(logger=logger,SchedulerConfig=SimpleNamespace(DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP=2048,DEFAULT_MAX_NUM_BATCHED_TOKENS=2048,DEFAULT_MAX_NUM_SEQS=256))
future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future,method],type_ignores=[])),str(SOURCE),'exec'),ns)
native=ns[method.name]
def make(name,mode='candidate',vendor='iluvatar',seq=None,budget=None):
    class Args:
        _set_default_max_num_seqs_and_batched_tokens_args=native
        @classmethod
        def get_batch_defaults(cls,world_size):
            return {Usage.OPENAI_API_SERVER:2048,Usage.LLM_CLASS:8192},{Usage.OPENAI_API_SERVER:256,Usage.LLM_CLASS:256}
    owner=Args();owner.__dict__.update(max_num_batched_tokens=budget,max_num_seqs=seq,performance_mode='balanced',
        enable_chunked_prefill=True,pipeline_parallel_size=1,tensor_parallel_size=1,speculative_config=None,enable_lora=False)
    model=SimpleNamespace(model='/workspace/MiniCPM5-2B',max_model_len=131072,dtype='torch.bfloat16',quantization=None,is_multimodal_model=False,is_mm_prefix_lm=False,
        hf_config=SimpleNamespace(hidden_size=2048,num_hidden_layers=42,num_attention_heads=16,num_key_value_heads=2))
    parallel=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,use_batched_dp_moe=False)
    module=SimpleNamespace(__file__=str(SOURCE),EngineArgs=Args,current_platform=SimpleNamespace(vendor_name=vendor))
    policy.budget.install(module,'candidate',root/name/'budget');policy.install(module,mode,root/name/'sequence')
    return owner,model,parallel
checks=[]
for name,kwargs,change,usage,expected_seq,expected_budget in [
    ('selected',{},None,Usage.OPENAI_API_SERVER,64,4096),
    ('control',dict(mode='control'),None,Usage.OPENAI_API_SERVER,256,4096),
    ('explicit64',dict(seq=64),None,Usage.OPENAI_API_SERVER,64,2048),
    ('explicit256',dict(seq=256),None,Usage.OPENAI_API_SERVER,256,4096),
    ('explicit32',dict(seq=32),None,Usage.OPENAI_API_SERVER,32,2048),
    ('explicit_budget8192',dict(budget=8192),None,Usage.OPENAI_API_SERVER,64,8192),
    ('other_vendor',dict(vendor='metax'),None,Usage.OPENAI_API_SERVER,256,2048),
    ('other_model',{},lambda m,p:setattr(m,'model','/other'),Usage.OPENAI_API_SERVER,256,2048),
    ('other_usage',{},None,Usage.LLM_CLASS,256,8192),
    ('parallel2',{},lambda m,p:setattr(p,'tensor_parallel_size',2),Usage.OPENAI_API_SERVER,256,2048)]:
    owner,model,parallel=make(name,**kwargs)
    if change:change(model,parallel)
    answer=owner._set_default_max_num_seqs_and_batched_tokens_args(usage,model,parallel)
    assert answer is None and owner.max_num_seqs==expected_seq and owner.max_num_batched_tokens==expected_budget,name
    budget_rows=json.loads(next((root/name/'budget').glob('*.json')).read_text());assert len(budget_rows)==1
    seq_rows=json.loads(next((root/name/'sequence').glob('*.json')).read_text());assert len(seq_rows)==1
    checks.append(dict(case=name,max_num_seqs=owner.max_num_seqs,budget=owner.max_num_batched_tokens,applied=seq_rows[0]['default_applied']))
sentinel=RuntimeError('native exception');before=logger.debug
try:
    logger.debug=lambda *a,**k:(_ for _ in ()).throw(sentinel)
    outcomes=[]
    for mode in ('control','candidate'):
        owner,model,parallel=make('exception_'+mode,mode=mode)
        try:owner._set_default_max_num_seqs_and_batched_tokens_args(Usage.OPENAI_API_SERVER,model,parallel)
        except RuntimeError as error:assert error is sentinel
        else:raise AssertionError('native exception swallowed')
        outcomes.append(dict(owner.__dict__))
    assert outcomes[0]==outcomes[1]
    checks.append(dict(case='native_exception_identity_and_state_equal'))
finally:logger.debug=before
sources=[*BASE.glob('*.py'),SOURCE,Path(policy.budget.__file__)]
result=dict(complete=True,CPU_only=True,GPU_executed=False,adopted=False,native_default_setting_AST_executed=True,
            guarded_limit=64,only_missing_native_default_changed=True,explicit_values_and_original_exceptions_preserved=True,
            budget4096_composition_calls_once=True,cases=checks,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            scope='No GPU or throughput claim; current S16 release remains unchanged')
for p in sources:
    target=root/'source_snapshot'/p.resolve().relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
(root/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(complete=True,CPU_only=True,cases=len(checks))))
