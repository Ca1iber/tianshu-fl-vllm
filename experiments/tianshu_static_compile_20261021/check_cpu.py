# SPDX-License-Identifier: Apache-2.0
"""Actual native Range/dispatch/config-copy AST; guarded default semantics."""
import ast,copy,dataclasses,importlib.util,json,sys,types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from candidate_profile import BASE,prepare,NATIVE
from policy import install,ARG_SHA,MODEL_SHA,sha,MODEL

def ast_member(path,cls,name):
    tree=ast.parse(Path(path).read_text());owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)
future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
utils=Path('/usr/local/lib/python3.12/site-packages/vllm/config/utils.py')
piece=Path('/usr/local/lib/python3.12/site-packages/vllm/compilation/piecewise_backend.py')
config=Path('/usr/local/lib/python3.12/site-packages/vllm/config/compilation.py')
args=Path('/usr/local/lib/python3.12/site-packages/vllm/engine/arg_utils.py')
assert sha(args)==ARG_SHA and sha(MODEL)==MODEL_SHA
native_range=next(n for n in ast.parse(utils.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Range')
native_entry=next(n for n in ast.parse(piece.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RangeEntry')
native_find=ast_member(piece,'PiecewiseBackend','_find_range_for_shape')
native_normalize=ast_member(config,'CompilationConfig','post_init_cudagraph_sizes')
engine=ast_member(args,'EngineArgs','create_engine_config')
assignments=[n for n in ast.walk(engine) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='compilation_config' for t in n.targets) and ast.unparse(n.value)=='copy.deepcopy(self.compilation_config)']
assert len(assignments)==1
module=types.ModuleType('_static_compile_actual_native_CPU_AST');sys.modules[module.__name__]=module
namespace=module.__dict__;namespace.update(dataclass=dataclasses.dataclass,dataclasses=dataclasses,Any=object)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future,native_range,native_entry,native_find,native_normalize],type_ignores=[])),str(piece),'exec'),namespace)
Range,Entry,find,normalize=[namespace[n] for n in ('Range','RangeEntry','_find_range_for_shape','post_init_cudagraph_sizes')]
checks=[]
for sizes in ([],[8,64]):
    dynamic=Range(1,4096);entries={dynamic:Entry(dynamic,compiled=True,runnable=lambda:None)}
    for m in sizes:entries[Range(m,m)]=Entry(Range(m,m),compiled=True,runnable=lambda:None)
    owner=SimpleNamespace(compile_sizes=sizes,compile_ranges=[dynamic],range_entries=entries)
    for m in range(0,4200):
        row=find(owner,m)
        assert row is (None if not 1<=m<=4096 else entries[Range(m,m) if m in sizes else dynamic])
    checks.append(dict(compile_sizes=sizes,native_dispatch_shapes=4200,exact_entries_have_priority=True,unlisted_shapes_retain_dynamic_range=True))
class Usage(Enum):OPENAI_API_SERVER='openai_api_server';LLM_CLASS='llm_class'
class Graph(Enum):FULL_DECODE_ONLY=1;FULL=2
raw=Path('/workspace/benchmark-results/20261021_static_compile_CPU_native_defaults_tianshu');assert not raw.exists();raw.mkdir()
root407=Path('/workspace/benchmark-results/20261021_prefill_int8_official4k_tianshu')
points=json.loads(next((root407/'bucket_trace_records').glob('configured_*.json')).read_text())['points'];assert len(points)==83 and max(points)==4096
copy_code=compile(ast.fix_missing_locations(ast.Module(body=[assignments[0]],type_ignores=[])),str(args),'exec')
guards=[]
def make(name,mode='candidate',sizes=None,vendor='iluvatar',usage=Usage.OPENAI_API_SERVER,mutate=None,fail=None):
    calls=[]
    class Args:
        def create_engine_config(self,usage_context=None,headless=False):
            calls.append(dict(usage=usage_context,headless=headless));local=dict(copy=copy,self=self);exec(copy_code,local)
            copied=local['compilation_config'];normalize(copied)
            if fail is not None:raise fail
            answer=SimpleNamespace(compilation_config=copied,scheduler_config=SimpleNamespace(max_num_batched_tokens=4096,max_num_seqs=64));calls[-1]['answer']=answer;return answer
    owner=Args();owner.__dict__.update(model='/workspace/MiniCPM5-2B',max_model_len=131072,dtype='auto',quantization=None,speculative_config=None,enable_lora=False,enforce_eager=False,tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,performance_mode='balanced',compilation_config=SimpleNamespace(compile_sizes=sizes,cudagraph_mode=Graph.FULL_DECODE_ONLY,cudagraph_capture_sizes=list(points),max_cudagraph_capture_size=4096))
    if mutate:mutate(owner)
    native=SimpleNamespace(__file__=str(args),EngineArgs=Args,current_platform=SimpleNamespace(vendor_name=vendor))
    install(native,mode,raw/name);before=copy.deepcopy(owner.__dict__);identity=owner.compilation_config.compile_sizes
    if fail is not None:
        try:owner.create_engine_config(usage_context=usage,headless=True)
        except RuntimeError as observed:assert observed is fail
        else:raise AssertionError('native error swallowed')
        assert owner.__dict__==before and owner.compilation_config.compile_sizes is identity and len(calls)==1;return
    answer=owner.create_engine_config(usage_context=usage,headless=True)
    assert answer is calls[0]['answer'] and calls[0]['usage'] is usage and calls[0]['headless'] is True and len(calls)==1
    assert owner.__dict__==before and owner.compilation_config.compile_sizes is identity
    row=json.loads(next((raw/name).glob('config_*.json')).read_text())[-1]
    assert row['explicit_compile_sizes_preserved'] and row['original_EngineArgs_compile_sizes_restored']
    guards.append(dict(name=name,actual_compile_sizes=answer.compilation_config.compile_sizes,applied=row['default_applied']))
    return answer,row
assert make('control',mode='control')[0].compilation_config.compile_sizes==[]
assert make('candidate')[0].compilation_config.compile_sizes==[8,64]
for name,sizes in (('explicit_empty',[]),('explicit_32',[32]),('explicit_unsorted',[64,8]),('explicit_capture_alias',['cudagraph_capture_sizes'])):
    answer,row=make(name,sizes=sizes);assert not row['default_applied'];expected=SimpleNamespace(compile_sizes=copy.deepcopy(sizes),cudagraph_capture_sizes=list(points),max_cudagraph_capture_size=4096);normalize(expected);assert answer.compilation_config.compile_sizes==expected.compile_sizes
for name,kw in (('other_vendor',dict(vendor='metax')),('LLM_usage',dict(usage=Usage.LLM_CLASS)),('no_usage',dict(usage=None)),('other_model',dict(mutate=lambda o:setattr(o,'model','/other'))),('FP16',dict(mutate=lambda o:setattr(o,'dtype','float16'))),('parallel2',dict(mutate=lambda o:setattr(o,'tensor_parallel_size',2))),('LoRA',dict(mutate=lambda o:setattr(o,'enable_lora',True))),('spec',dict(mutate=lambda o:setattr(o,'speculative_config',{}))),('eager',dict(mutate=lambda o:setattr(o,'enforce_eager',True))),('other_graph',dict(mutate=lambda o:setattr(o.compilation_config,'cudagraph_mode',Graph.FULL))),('other_max_len',dict(mutate=lambda o:setattr(o,'max_model_len',32768)))):
    answer,row=make(name,**kw);assert not row['default_applied'] and answer.compilation_config.compile_sizes==[]
make('native_exception_candidate',fail=RuntimeError('native sentinel'));make('native_exception_control',mode='control',fail=RuntimeError('native control sentinel'))
for p in BASE.rglob('*.py'):ast.parse(p.read_text())
profiles=[]
for mode in ('control','candidate'):
    target=Path(f'/workspace/benchmark-results/20261021_static_compile_CPU_profile_{mode}_tianshu');assert not target.exists();target.mkdir()
    command,env,settings,sources=prepare(target,mode);assert len(command)==13 and len(settings)==62
    assert settings['PYTHONPATH']==str(BASE/('hooks_'+mode))
    profiles.append(dict(mode=mode,command=command,profile=settings,sources=sources))
assert profiles[0]['sources']==profiles[1]['sources']
proof=dict(complete=True,CPU_only=True,no_GPU_started=True,actual_native_Range_and_find_AST_executed=True,actual_native_config_deepcopy_and_normalization_AST_executed=True,
    native_dispatch_checks=checks,guard_cases=guards,native_exception_identity_and_input_restoration_verified=True,all_AST_parsed=True,
    original_serve13argv_and62environment_preserved=True,parent_runtime_sources=407,runtime_source_count=len(profiles[0]['sources']),environment_count=62,new_Math_required_if_adopted=True,profiles=profiles)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='profiles'}))
