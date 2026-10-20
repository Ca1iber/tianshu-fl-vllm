# SPDX-License-Identifier: Apache-2.0
"""Execute native admission method with a CPU fake coordinator; no scheduler/performance simulation."""
import ast,hashlib,importlib.util,json,math,sys,tempfile
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
BASE=Path(__file__).resolve().parent
SOURCE=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py')
SCHEDULER=SOURCE.parent/'sched/scheduler.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
node=next(x for x in ast.parse(SOURCE.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='KVCacheManager')
method=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=='allocate_slots')
class Status(Enum):WAITING=1;PREEMPTED=2;RUNNING=3
ns=dict(RequestStatus=Status)
future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future,method],type_ignores=[])),str(SOURCE),'exec'),ns)
native=ns['allocate_slots']
class Coordinator:
    def __init__(self):self.allocations=[]
    def get_num_blocks_to_allocate(self,**kwargs):return math.ceil(kwargs['num_tokens']/16)
    def remove_skipped_blocks(self,*args):pass
    def allocate_new_blocks(self,*args):self.allocations.append(args);return ('allocated',)

def check(name,*,status=Status.WAITING,scheduled=True,full=True,free=3673,reserved=0,expected):
    coordinator=Coordinator()
    owner=SimpleNamespace(empty_kv_cache_blocks=SimpleNamespace(blocks=((),)),max_model_len=131072,
        watermark_blocks=2649,coordinator=coordinator,block_pool=SimpleNamespace(get_num_free_blocks=lambda:free),
        enable_caching=False,create_kv_cache_blocks=lambda x:x)
    request=SimpleNamespace(request_id=name,num_computed_tokens=0,num_tokens=16384,status=status)
    answer=native(owner,request,4096,full_sequence_must_fit=full,reserved_blocks=reserved,has_scheduled_reqs=scheduled)
    assert (answer is not None)==expected,name
    assert bool(coordinator.allocations)==expected
    return dict(name=name,status=status.name,has_scheduled_requests=scheduled,full_sequence_must_fit=full,
                free_blocks=free,reserved_blocks=reserved,admitted=expected)
checks=[check('waiting_exact_full_gate',expected=True),check('waiting_below_full_gate',free=3672,expected=False),
        check('preempted_below_full_gate',status=Status.PREEMPTED,free=3672,expected=False),
        check('running_not_gated_by_watermark',status=Status.RUNNING,free=1024,expected=True),
        check('first_waiting_can_progress_without_watermark',scheduled=False,free=1024,expected=True),
        check('per_step_exact_watermark_gate',full=False,free=2905,expected=True),
        check('per_step_below_watermark_gate',full=False,free=2904,expected=False),
        check('other_inflight_reserved_blocks_kept',reserved=900,expected=False)]
# Preserve the actual native method, forwarded sampling/model settings and all parent hook installation.
for path in BASE.rglob('*.py'):ast.parse(path.read_text())
old=BASE.parent/'tianshu_long_segments_service_20261020'
assert (BASE/'base_runtime_audit.py').read_text().replace('== 48','== 46')==(old/'base_runtime_audit.py').read_text()
assert (BASE/'long_runtime_audit.py').read_bytes()==(old/'audit_profile.py').read_bytes()
assert (BASE/'run_service_pair.py').read_text().replace("key=='TIANSHU_KV_HEADROOM_MODE'","key=='TIANSHU_LONG_SEGMENT_MODE'")==(old/'run_service_pair.py').read_text()
assert (BASE/'run_service_trial.py').read_bytes()==(old/'run_service_trial.py').read_bytes()
hook=ast.parse((BASE/'hooks/sitecustomize.py').read_text())
calls=[x for x in ast.walk(hook) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='runpy' and x.func.attr=='run_path']
assert len(calls)==1 and 'tianshu_long_segments_service_20261020/hooks/sitecustomize.py' in ast.unparse(calls[0])
legacy_gate=BASE.parent/'tianshu_kv_headroom_20261020/CPU_GATE_32g.json'
gate=json.loads(legacy_gate.read_text());assert gate['complete']
for name,digest in gate['frozen_sources'].items():assert sha(name)==digest,name
sources=[*BASE.rglob('*.py'),BASE/'batch_policy.json',legacy_gate,legacy_gate.with_name('policy.py'),SOURCE,SCHEDULER]
result=dict(complete=True,CPU_only=True,GPU_executed=False,adopted=False,native_admission_AST_executed=True,
            coordinator_is_CPU_test_double=True,not_a_full_scheduler_simulation=True,no_throughput_prediction=True,
            both_arms_use_identical_long_segment_candidate=True,parent_hook_installed_once=True,
            only_runtime_configuration_difference='native watermark 0 versus 1/16',guard_cases=checks,
            sources={str(p):sha(p) for p in sources})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
s=importlib.util.spec_from_file_location('_headroom_service_cpu_profile',BASE/'candidate_profile.py')
profile=importlib.util.module_from_spec(s);s.loader.exec_module(profile)
with tempfile.TemporaryDirectory(dir='/workspace/benchmark-results',prefix='headroom_service_CPU_') as temporary:
    root=Path(temporary)
    ca,ea,pa,sa=profile.prepare(root/'control','control')
    cb,eb,pb,sb=profile.prepare(root/'candidate','candidate')
    assert ca==cb and sa==sb
    for key in pa:
        if key=='TIANSHU_KV_HEADROOM_MODE':assert (pa[key],pb[key])==('control','candidate')
        else:assert pa[key].replace(str(root/'control'),'<RUN>')==pb[key].replace(str(root/'candidate'),'<RUN>'),key
    assert len(pa)==48 and pa['TIANSHU_LONG_SEGMENT_MODE']==pb['TIANSHU_LONG_SEGMENT_MODE']=='candidate'
    result.update(profile_source_count=len(sa),profile_environment_count=len(pa),both_original_serve_argv_identical=True,
                  both_frozen_source_sets_identical=True,normalized_environment_diff_exactly_watermark_mode=True)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('sources','guard_cases')}))
