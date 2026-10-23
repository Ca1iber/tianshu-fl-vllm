# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SERVICE=EXP/'tianshu_W4_LMpad_service_v2_20261023';HOLDOUT=EXP/'tianshu_native_int8_attention42_20261022'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
parent=load('_LMpad_v2_model_profile',SERVICE/'candidate_profile.py');down=load('_LMpad_v2_preserve_Down_and_r3_audits',EXP/'tianshu_dotsaddz_Down64_quality_v2_20261023/quality_profile.py');own=load('_LMpad_v2_own_actual_checks',SERVICE/'audit_profile.py')
adapted_C8=load('_LMpad_v2_C8_precision_checks',SERVICE/'inherited_C8_dense_audit.py');adapted_C64=load('_LMpad_v2_C64_precision_checks',SERVICE/'inherited_C64_dense_audit.py');adapted_Down=load('_LMpad_v2_Down_precision_checks',SERVICE/'inherited_down_audit.py')
original_C8_parent=down.core.q8.parent.parent;original_C64_parent=down.core.q64.parent.parent
adapted_C8.parent=original_C8_parent;adapted_C64.parent=original_C64_parent;down.core.q8.parent=adapted_C8;down.core.q64.parent=adapted_C64;down.own=adapted_Down
# Keep actual native C64 admission through original shared baseline module.
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check;BENCH_SHA,csv_rows,normalized_environment=down.BENCH_SHA,down.csv_rows,down.normalized_environment

def prepare(root,screen_root,batch=8,scope='prefill'):
 assert screen_root.resolve()==HOLDOUT.resolve() and batch in (8,64) and scope=='prefill';command,env,profile,sources=parent.prepare(root,'candidate');assert len(sources)==read(SERVICE/'CPU_GATE_32g.json')['expected_model_source_count']
 if batch==64:assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate';profile['TIANSHU_FUTURE_KV_MODE']='control';env.update(profile)
 previous=read(Path('/workspace/benchmark-results/20261023_dots_Down64_v2_math8_tianshu')/'controller_sources.json');paths={Path(p) for p in previous}|set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_quality.sh'}
 for name in ('EXECUTION.sha256','CPU_GATE_32g.json'):
  if (BASE/name).exists():paths.add(BASE/name)
 for module in tuple(sys.modules.values()):
  p=getattr(module,'__file__',None)
  if p and p.endswith('.py') and Path(p).is_relative_to(EXP):paths.add(Path(p).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,env,profile,sources

def audit_startup(root,state):
 result=down.audit_startup(root,state);result.update(own.audit_startup(root,state));return result

def audit_runtime(root,state,official=False):
 assert not official;result=down.audit_runtime(root,state,official=False);result.update(own.audit_runtime(root,state,minimum_calls=10,batch=state['quality_batch']))
 assert result['verified_dense_batch']==state['quality_batch'] and result['LM_head_precision_in_verified_batch']=='W4A16'
 result.update(original_GU_KV_attention_scheduler_sampling_LM_checks_retained=False,all_original_GU_KV_attention_scheduler_sampling_checks_retained=True,only_original_LM1_64_precision_requirements_explicitly_replaced=True,LM8_old_BF16_requirement_replaced_by_actual_W4=True,LM64_old_W8_requirement_replaced_by_actual_W4=True)
 return result

def audit_math(root,state,profile,write=True):
 result=down.core.original_math(root,state,profile,write=False);batch=state['quality_batch'];assert profile['verified_dense_batch']==batch and profile['LM_head_precision_in_verified_batch']=='W4A16' and profile['new_LM_verified_batch_W4_runtime_calls']>=10
 assert profile['old_LM1_64_BF16_W8_calls_absent'] and profile['new_LM1_64_range_and_padding_profile_verified']
 assert profile['actual_new_prefill_path_all_42_GateUp_verified'] or (batch==8 and profile['actual_BF16_prefill_delegation_all42_verified'])
 assert profile['actual_Down64_W4_capture_banks']==42 and profile['old_Down64_BF16_capture_banks']==0 and profile['dots_Down64_load_and_native_memory_scratch_all42_accounted'] and profile['new_LM64_actual_W4_bank_verified'] and profile['new_LM64_load_and_native_peak_accounting_verified']
 assert profile['actual_attention_M64_capture_native_banks']==0 and profile['new_Down_memory_profile_all42_accounted'] and profile['new_attention_memory_profile_all84_accounted']
 if batch==64:assert profile['actual_C64_fused_replay_verified'] and profile['actual_native_prefill_bank_count']==42 and 4096 in profile['actual_native_prefill_M_values'] and profile['new_LM64_W4_actual_runtime_calls']>=10 and profile['dots_Down64_captured_replay_linked_to_inherited_model_M64_execution']
 result.update(quality_batch=batch,fresh_math_for_original_command_profile=batch==8,supplementary_C64_quality=batch==64,controller_sources=state['controller_sources'],additional_environment=state['additional_environment'],runtime_source_count=len(state['frozen_sources']),environment_count=63,previous_accuracy_not_reused=True,only_new_precision_vs_r3='M64_Down_W4_and_LM_M1_to64_W4_direct1_2_or_pad32_64',only_new_precision_vs_934_model='LM_M1_to63_W4',actual_native_prefill_bank_count=profile['actual_native_prefill_bank_count'],actual_native_prefill_M_values=profile['actual_native_prefill_M_values'],actual_Down64_W4_capture_banks=42,new_LM64_W4_actual_runtime_calls=profile['new_LM64_W4_actual_runtime_calls'],new_LM8_W4_actual_runtime_calls=profile['new_LM8_W4_actual_runtime_calls'],new_LM_W4_calls_by_M=profile['new_LM_W4_calls_by_M'],new_LM64_payload_bytes=142049280,no_original_parent_counters_modified=True,old_LM1_64_BF16_and_W8_requirements_explicitly_replaced=True)
 if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def audit_benchmark(*args,**kwargs):raise AssertionError('Fresh entire C8/C64 Math required before model performance')
