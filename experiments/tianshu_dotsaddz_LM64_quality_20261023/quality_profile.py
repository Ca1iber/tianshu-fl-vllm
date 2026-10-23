# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SERVICE=EXP/'tianshu_dotsaddz_LM64_service_20261023';HOLDOUT=EXP/'tianshu_native_int8_attention42_20261022'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
parent=load('_LM64_quality_service',SERVICE/'candidate_profile.py');down=load('_LM64_quality_full_Down_audits',EXP/'tianshu_dotsaddz_Down64_quality_v2_20261023/quality_profile.py');own=load('_LM64_quality_explicit_new_checks',SERVICE/'audit_profile.py');adapted_C64=load('_LM64_quality_explicit_C64_guard_replacement',SERVICE/'inherited_C64_dense_audit.py')
# Only the old LM64 INT8 requirement is replaced; preserve native admission and all GU checks.
adapted_C64.parent=down.core.q64.parent.parent;down.core.q64.parent=adapted_C64
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check;BENCH_SHA,csv_rows,normalized_environment=down.BENCH_SHA,down.csv_rows,down.normalized_environment

def prepare(root,screen_root,batch=8,scope='prefill'):
 assert screen_root.resolve()==HOLDOUT.resolve() and batch in (8,64) and scope=='prefill';command,env,profile,sources=parent.prepare(root,'candidate');assert len(sources)==934
 if batch==64:assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate';profile['TIANSHU_FUTURE_KV_MODE']='control';env.update(profile)
 previous=read(Path('/workspace/benchmark-results/20261023_dots_Down64_v2_math8_tianshu')/'controller_sources.json');paths={Path(p) for p in previous}|set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_quality.sh'}
 for name in ('EXECUTION.sha256','CPU_GATE_32g.json'):
  if (BASE/name).exists():paths.add(BASE/name)
 for m in tuple(sys.modules.values()):
  p=getattr(m,'__file__',None)
  if p and p.endswith('.py') and Path(p).is_relative_to(EXP):paths.add(Path(p).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,env,profile,sources

def audit_startup(root,state):
 result=down.audit_startup(root,state);result.update(own.audit_startup(root,state));return result

def audit_runtime(root,state,official=False):
 assert not official;result=down.audit_runtime(root,state,official=False);result.update(own.audit_runtime(root,state,minimum_calls=10 if state['quality_batch']==64 else 0));result.update(original_LM64_INT8_precision_retained=False,LM64_guard_explicitly_replaced_with_real_W4_execution=True,all_original_LM8_GU_KV_scheduler_sampling_attention_and_Down_checks_retained=True);return result

def audit_math(root,state,profile,write=True):
 result=down.core.original_math(root,state,profile,write=False);batch=state['quality_batch'];assert profile['verified_dense_batch']==batch and profile['LM_head_precision_in_verified_batch']==('BF16' if batch==8 else 'W4A16')
 assert profile['actual_new_prefill_path_all_42_GateUp_verified'] or (batch==8 and profile['actual_BF16_prefill_delegation_all42_verified'])
 assert profile['actual_Down64_W4_capture_banks']==42 and profile['old_Down64_BF16_capture_banks']==0 and profile['dots_Down64_load_and_native_memory_scratch_all42_accounted'] and profile['new_LM64_actual_W4_bank_verified'] and profile['new_LM64_load_and_native_peak_accounting_verified']
 assert profile['actual_attention_M64_capture_native_banks']==0 and profile['new_Down_memory_profile_all42_accounted'] and profile['new_attention_memory_profile_all84_accounted']
 if batch==64:assert profile['actual_C64_fused_replay_verified'] and profile['actual_native_prefill_bank_count']==42 and 4096 in profile['actual_native_prefill_M_values'] and profile['new_LM64_W4_actual_runtime_calls']>=10 and profile['dots_Down64_captured_replay_linked_to_inherited_model_M64_execution']
 result.update(quality_batch=batch,fresh_math_for_original_command_profile=batch==8,supplementary_C64_quality=batch==64,controller_sources=state['controller_sources'],additional_environment=state['additional_environment'],runtime_source_count=934,environment_count=63,previous_accuracy_not_reused=True,only_new_precision_vs_r3='M64_Down_W4_and_M64_LM_W4',only_new_precision_vs_Down64_parent='M64_LM_group128_RTN_W4A16_dotsaddz',actual_native_prefill_bank_count=profile['actual_native_prefill_bank_count'],actual_native_prefill_M_values=profile['actual_native_prefill_M_values'],actual_Down64_W4_capture_banks=42,old_Down64_BF16_capture_banks=0,new_LM64_W4_actual_runtime_calls=profile['new_LM64_W4_actual_runtime_calls'],new_LM64_payload_bytes=142049280,original_LM64_W8_requirement_explicitly_replaced=True,no_original_parent_counters_modified=True,C8_LM64_runtime_coverage_not_claimed=batch==8)
 if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def audit_benchmark(*args,**kwargs):raise AssertionError('Fresh complete Math required before new performance tests')
