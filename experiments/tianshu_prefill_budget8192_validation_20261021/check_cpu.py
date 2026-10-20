# SPDX-License-Identifier: Apache-2.0
"""Freeze quality controllers; exercise C8/C64 audits on existing screen only."""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
SERVICE=BASE.parent/'tianshu_prefill_budget8192_service_20261021'
SCREEN=BASE.parent/'tianshu_prefill_budget8192_audit_results_20261021'
OLD=BASE.parent/'tianshu_dense_budget_validation_v2_20261021'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
assert sha(BASE/'baseline_quality_audit.py')==sha(OLD/'baseline_quality_audit.py')
expected=(OLD/'audit_profile.py').read_text()
changes=[('    assert len({r["prefix"] for r in actual})==42 and all(r["implementation"]=="linear_contiguous" and r["calls"]>0 for r in actual)','    assert all(r["implementation"]=="linear_contiguous" and r["calls"]>0 for r in actual)'),
('actual_new_prefill_path_all_42_GateUp_verified=True','actual_BF16_fallback_prefill_prefixes=len({r["prefix"] for r in actual})'),
('runtime_source_count=374','runtime_source_count=445'),
("row[prefix+'_sha256']",'row["native_source_sha256" if prefix=="native" else "policy_sha256"]')]
for before,after in changes:assert expected.count(before)==1,before;expected=expected.replace(before,after)
assert expected==(BASE/'dense_quality_audit.py').read_text()
for p in BASE.glob('*.py'):ast.parse(p.read_text())
profile=read(SERVICE/'CPU_GATE_32g.json');assert profile['complete'] and profile['runtime_source_count']==445 and profile['environment_count']==64
runtime=profile['profiles'][1]['frozen_sources'];assert all(sha(p)==v for p,v in runtime.items())
gate=read(SCREEN/'SCREEN_GATE_32g.json');assert gate['complete'] and gate['independently_rechecked'] and gate['metadata_only_recovery'] and gate['gain_percent']>=1.
root=Path(gate['raw_roots']['candidate']);state=read(root/'result.json')
assert sha(root/'result.json')==gate['raw_result_sha256']['candidate'] and state['frozen_sources']==runtime
quality_audit=load('_new8192_quality_audit_CPU_probe',BASE/'audit_profile.py')
checks={}
for official in (False,True):
    answer=quality_audit.audit_runtime(root,dict(state,seq_limit=64,arena_mode='candidate',stage='screen_CPU_preflight'),official=official)
    assert answer['scheduler_budget']==8192 and answer['actual_KV_blocks']==40219 and answer['actual_new_prefill_path_all_42_GateUp_verified']
    assert answer['verified_dense_batch']==(64 if official else 8)
    checks['C64_screen' if official else 'C8_untimed_canary']=dict(actual_runtime_audit_passed=True,verified_dense_batch=answer['verified_dense_batch'],prefill_INT8_shapes=answer['actual_prefill_INT8_shapes'])
assert quality_audit.parent.parent.base.__file__.endswith('tianshu_dense_budget8192_v2_20261021/base_runtime_audit.py')
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'batch_policy.json',SERVICE/'CPU_GATE_32g.json',BASE.parent/'tianshu_original_command_20261016/validation_profile.py',SCREEN/'SCREEN_GATE_32g.json',SCREEN/'SCREEN_ROUNDS_32g.json',SCREEN/'recovered_runtime_audit.py',SCREEN/'verify_pair.py',SCREEN/'CONTROL_RECOVERY_GATE_32g.json',SCREEN/'CONTROLLERS_32g.sha256']
proof=dict(complete=True,CPU_only=True,no_GPU_started=True,new_Math_not_run=True,new_official_benchmark_not_run=True,runtime_source_count=445,environment_count=64,
    actual_screen_C8_canary_and_C64_audit_preflight=checks,screen_canary_not_Math_quality=True,baseline_quality_byte_identical_to_existing_budget8192=True,explicit_dense_quality_adaptations=4,
    runtime_sources_unchanged=runtime,sources={str(p.resolve()):sha(p) for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(proof,indent=2)+'\n')
framework=load('_new8192_validation_framework_CPU_probe',BASE/'framework_profile.py')
dry=Path('/workspace/benchmark-results/20261021_prefill_budget8192_validation_CPU_dry_tianshu');assert not dry.exists();dry.mkdir()
command,env,settings,actual=framework.prepare(dry,SCREEN)
assert len(command)==13 and len(settings)==64 and actual==runtime and settings['TIANSHU_DENSE_BUDGET_MODE']=='candidate'
controllers=read(dry/'controller_sources.json');assert all(sha(p)==v for p,v in controllers.items())
preflight=dict(complete=True,CPU_only=True,no_GPU_started=True,original13argv_and_screened445source64env_preserved=True,primary_controller_count=len(controllers),CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),controller_sources=controllers)
(BASE/'CPU_PREFLIGHT_GATE_32g.json').write_text(json.dumps(preflight,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k not in ('sources','runtime_sources_unchanged')}));print(json.dumps({k:v for k,v in preflight.items() if k!='controller_sources'}))
