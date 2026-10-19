# SPDX-License-Identifier: Apache-2.0
"""Execute the installed native default-setting AST and test guarded selection."""
import argparse,ast,hashlib,json,tempfile
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from default_policy import ARG_UTILS_SHA,install
p=argparse.ArgumentParser();p.add_argument('--arg-utils-source',type=Path,required=True);a=p.parse_args()
assert hashlib.sha256(a.arg_utils_source.read_bytes()).hexdigest()==ARG_UTILS_SHA
tree=ast.parse(a.arg_utils_source.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='EngineArgs')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_set_default_max_num_seqs_and_batched_tokens_args')
class Usage(Enum):OPENAI_API_SERVER='openai_api_server';LLM_CLASS='llm_class'
logger=SimpleNamespace(debug=lambda *a,**k:None,info=lambda *a,**k:None)
ns=dict(logger=logger,SchedulerConfig=SimpleNamespace(DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP=2048,DEFAULT_MAX_NUM_BATCHED_TOKENS=2048,DEFAULT_MAX_NUM_SEQS=256))
future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future,method],type_ignores=[])),str(a.arg_utils_source),'exec'),ns)
native=ns[method.name]


def make(mode='candidate',vendor='iluvatar',budget=None,performance='balanced',chunked=True,seqs=None,install_wrapper=True):
    class Args:
        _set_default_max_num_seqs_and_batched_tokens_args=native
        @classmethod
        def get_batch_defaults(cls,world_size):
            return {Usage.OPENAI_API_SERVER:2048,Usage.LLM_CLASS:8192},{Usage.OPENAI_API_SERVER:256,Usage.LLM_CLASS:256}
    owner=Args();owner.__dict__.update(max_num_batched_tokens=budget,max_num_seqs=seqs,performance_mode=performance,enable_chunked_prefill=chunked,pipeline_parallel_size=1,tensor_parallel_size=1,speculative_config=None,enable_lora=False)
    model=SimpleNamespace(model='/workspace/MiniCPM5-2B',max_model_len=131072,dtype='torch.bfloat16',quantization=None,is_multimodal_model=False,is_mm_prefix_lm=False,hf_config=SimpleNamespace(hidden_size=2048,num_hidden_layers=42,num_attention_heads=16,num_key_value_heads=2))
    parallel=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,use_batched_dp_moe=False)
    directory=Path(tempfile.mkdtemp(prefix='tianshu-default-cpu-'))
    module=SimpleNamespace(__file__=str(a.arg_utils_source),EngineArgs=Args,current_platform=SimpleNamespace(vendor_name=vendor))
    if install_wrapper:install(module,mode,directory)
    return owner,model,parallel,directory


checks=[]
for name,kwargs,change,expected,applied in (
 ('control',dict(mode='control'),None,2048,False),
 ('selected',{},None,4096,True),
 ('explicit2048',dict(budget=2048),None,2048,False),
 ('explicit8192',dict(budget=8192),None,8192,False),
 ('other_vendor',dict(vendor='metax'),None,2048,False),
 ('other_model',{},lambda m,p:setattr(m,'model','/other'),2048,False),
 ('other_geometry',{},lambda m,p:setattr(m.hf_config,'hidden_size',4096),2048,False),
 ('FP16',{},lambda m,p:setattr(m,'dtype','torch.float16'),2048,False),
 ('parallel2',{},lambda m,p:setattr(p,'tensor_parallel_size',2),2048,False),
 ('throughput',dict(performance='throughput'),None,4096,False),
 ('chunked_off',dict(chunked=False),None,131072,False),
 ('explicit512seqs',dict(seqs=512),None,2048,False)):
    owner,model,parallel,directory=make(**kwargs)
    if change:change(model,parallel)
    assert owner._set_default_max_num_seqs_and_batched_tokens_args(usage_context=Usage.OPENAI_API_SERVER,model_config=model,parallel_config=parallel) is None
    assert owner.max_num_batched_tokens==expected,name
    row=json.loads(next(directory.glob('*.json')).read_text())[-1];assert row['default_applied']==applied,name
    checks.append(name)
selected,model,parallel,_=make();selected._set_default_max_num_seqs_and_batched_tokens_args(Usage.OPENAI_API_SERVER,model,parallel)
explicit,m,p,_=make(budget=4096,install_wrapper=False);explicit._set_default_max_num_seqs_and_batched_tokens_args(Usage.OPENAI_API_SERVER,m,p)
assert selected.__dict__==explicit.__dict__
owner,model,parallel,_=make();error=RuntimeError('native boundary failure')
def fail(*a,**k):raise error
previous=logger.debug;logger.debug=fail
try:
    try:owner._set_default_max_num_seqs_and_batched_tokens_args(Usage.OPENAI_API_SERVER,model,parallel)
    except RuntimeError as observed:assert observed is error
    else:raise AssertionError('exception swallowed')
    assert owner.max_num_batched_tokens is None
finally:logger.debug=previous
b=Path(__file__).resolve().parent
for p in b.rglob('*.py'):ast.parse(p.read_text())
r=dict(complete=True,CPU_only=True,actual_installed_default_setting_AST_executed=True,source_sha256=ARG_UTILS_SHA,guard_cases=checks,selected_owner_state_exactly_matches_explicit4096=True,original_exception_identity_and_budget_rollback_checked=True,GPU_not_run=True,new_service_and_Math_not_run=True,adopted=False,sources={str(p.relative_to(b)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(b.rglob('*.py'))})
(b/'CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
