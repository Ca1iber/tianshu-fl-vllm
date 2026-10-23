# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SERVICE=EXP/'tianshu_RMS_LM_service_20261023';HOLDOUT=EXP/'tianshu_LM_pad64_20261022'
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
parent=load('_RMS_LM_new_service',SERVICE/'candidate_profile.py');legacy=load('_RMS_LM_retained_RMS_r3_quality',EXP/'tianshu_RMS_A8_GU_quality_v2_20261023/quality_profile.py');lm=load('_RMS_LM_explicit_LM_audit',EXP/'tianshu_LM_pad64_service_20261022/audit_profile.py');repair=load('_RMS_LM_all_LM_guard_copies',EXP/'tianshu_LM_pad64_audit_recovery_20261022/patch_runtime.py');PATCHED=dict(core=lm.patch_dense_modules(legacy),math=repair.install(legacy,'math'),official=repair.install(legacy,'official'));digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check;BENCH_SHA,csv_rows,normalized_environment=legacy.BENCH_SHA,legacy.csv_rows,legacy.normalized_environment

def parent_quality_gate():
 gate=read(EXP/'tianshu_RMS_A8_GU_quality_results_v2_20261023/QUALITY_GATE_32g.json');assert gate['complete'] and gate['model_quality_qualified'] and gate['C8_correct']/105>.95 and gate['C64_correct']/105>.95 and gate['model_source_count']==916 and gate['controller_count']==80 and gate['all_inherited_and_new_runtime_audits_independently_recomputed'];frozen_check(gate['model_frozen_sources'])
 for p,h in gate['raw_files'].items():assert digest(p)==h
 return gate

def prepare(root,screen_root,batch=8,scope='prefill'):
 assert screen_root.resolve()==HOLDOUT.resolve() and batch in (8,64) and scope=='prefill';prior=parent_quality_gate();component=read(HOLDOUT/'INDEPENDENT_CPU_GATE_32g.json');assert component['complete'] and component['dynamic_range_lead'] and component['raw_result_sha256']==digest('/workspace/benchmark-results/20261022_LM_pad64_GPU_tianshu/result.json');command,environment,profile,sources=parent.prepare(root,'candidate');assert len(sources)==944 and all(sources[p]==h for p,h in prior['model_frozen_sources'].items())
 if batch==64:assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate';profile['TIANSHU_FUTURE_KV_MODE']='control';environment.update(profile)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_quality.sh',BASE/'LM_COMPONENT_CACHE.sha256',EXP/'tianshu_RMS_A8_GU_quality_results_v2_20261023/QUALITY_GATE_32g.json',EXP/'tianshu_RMS_A8_GU_quality_results_v2_20261023/verify_cpu.py',EXP/'tianshu_RMS_A8_GU_quality_results_v2_20261023/PLAN_32g.md'};release=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');paths.update(Path(p) for p in release['controller_sources'])
 for m in tuple(sys.modules.values()):
  p=getattr(m,'__file__',None)
  if p and p.endswith('.py') and Path(p).is_relative_to(EXP):paths.add(Path(p).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  t=root/'controller_source_snapshot'/Path(p).relative_to('/');t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(Path(p).read_bytes());assert digest(t)==h
 return command,environment,profile,sources

def audit_startup(root,state):
 result=legacy.audit_startup(root,state);result.update(lm.audit_startup(root,state));return result

def audit_runtime(root,state,official=False):
 assert not official;result=legacy.audit_runtime(root,state,official=False);result.update(lm.audit(root,state));result.update(lm.audit_startup(root,state));result.update(explicit_LM_guard_copies_replaced_before_first_GPU=PATCHED,old_LM64_execution_not_claimed=True,LM_precision_and_route_unchanged=False,RMS_record_LM_unchanged_flag_scope='RMS hook only; combined LM route is separately replaced and audited',all_other_r3_checks_and_RMS_fusion_retained=True);return result

def audit_math(root,state,profile,write=True):
 result=legacy.audit_math(root,state,profile,write=False);assert profile['new_LM_CUBIN_exact'] and profile['new_LM_original_pack_same'] and profile['new_LM_profile_peak_accounted'] and profile['legacy_LM64_guard_explicitly_replaced_by_new_integer_counter'];result.update(new_combined_RMS_LM_model=True,parent_model_scores_not_reused=True,new_LM_actual_runtime_M=profile['new_LM_actual_runtime_M'],new_LM_runtime_calls=profile['new_LM_runtime_calls'],new_LM64_original_bits_exact=True,new_LM_minimum_M=32,LM_precision_and_route_unchanged=False,old_LM64_execution_not_claimed=True,explicit_all_LM64_guard_copies_replaced_before_first_GPU=True)
 if state['quality_batch']==64:assert profile['new_LM_runtime_calls']>0 and 64 in profile['new_LM_actual_runtime_M']
 if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def audit_benchmark(*args,**kwargs):raise AssertionError('Combined fresh Math-only controller; future performance needs a separate frozen controller')
