# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SERVICE=EXP/'tianshu_dotsaddz_Down64_service_v2_20261023';HOLDOUT=EXP/'tianshu_native_int8_attention42_20261022'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
parent=load('_dots_Down64_quality_service',SERVICE/'candidate_profile.py');r3=load('_dots_Down64_all_original_r3_audits',EXP/'tianshu_native_int8_attention_quality_20261022/quality_profile.py');core=r3.legacy.legacy;down=load('_dots_Down64_explicit_old_capture_replacement',SERVICE/'inherited_down_audit.py');own=load('_dots_Down64_new_checks',SERVICE/'audit_profile.py');digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check;BENCH_SHA,csv_rows,normalized_environment=r3.BENCH_SHA,r3.csv_rows,r3.normalized_environment

def prepare(root,screen_root,batch=8,scope='prefill'):
 assert screen_root.resolve()==HOLDOUT.resolve() and batch in (8,64) and scope=='prefill';command,env,profile,sources=parent.prepare(root,'candidate');gate=read(SERVICE/'CPU_GATE_32g.json');assert len(sources)==gate['expected_model_source_count']==916
 registration=read(BASE/'REGISTRATION_CPU_GATE_32g.json');assert registration['complete'] and registration['public_register_kernel_cpu_dynamic_opaque_op_test_passed'] and not registration['CUDA_initialized'] and registration['registry_sha256']==digest(registration['registry_source'])
 if batch==64:assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate';profile['TIANSHU_FUTURE_KV_MODE']='control';env.update(profile)
 release=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');paths={Path(p) for p in release['controller_sources']}|set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_quality.sh',BASE/'REGISTRATION_CPU_GATE_32g.json',BASE/'CPU_PREPARATION_32g.md',BASE/'INITIAL_REGISTRATION_CPU.sha256'}
 for module in tuple(sys.modules.values()):
  p=getattr(module,'__file__',None)
  if p and p.endswith('.py') and Path(p).is_relative_to(EXP):paths.add(Path(p).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,env,profile,sources

def audit_startup(root,state):
 result=down.audit_startup(root,state);result.update(r3.own.audit_startup(root,state));result.update(own.audit_startup(root,state));return result

def audit_runtime(root,state,official=False):
 assert not official and state['down_mode']==state['attention_mode']=='candidate' and state['down_scope']=='prefill';result=core.audit_runtime(root,state,official=False);result.update(down.audit(root,state));result.update(r3.own.audit(root,state));result.update(audit_startup(root,state));result.update(own.audit(root,state,result));result.update(original903_r3_sources_preserved=True,all_original_Down_checks_except_M64_BF16_capture_retained=True,original_M64_BF16_requirement_replaced_by_42_real_W4_captures=True,original_GU_KV_attention_scheduler_sampling_LM_checks_retained=True);return result

def audit_math(root,state,profile,write=True):
 result=core.audit_math(root,state,profile,write=False);batch=state['quality_batch'];assert profile['all_original_Down_checks_except_M64_BF16_capture_retained'] and profile['original_M64_BF16_requirement_replaced_by_42_real_W4_captures'] and profile['original_GU_KV_attention_scheduler_sampling_LM_checks_retained'];assert profile['new_Down_memory_profile_all42_accounted'] and profile['new_attention_memory_profile_all84_accounted'] and profile['actual_attention_M64_capture_native_banks']==0;assert profile['actual_Down64_W4_capture_banks']==42 and profile['old_Down64_BF16_capture_banks']==0 and profile['dots_Down64_load_and_native_memory_scratch_all42_accounted']
 if batch==64:assert profile['actual_native_Down_runtime_banks']==42 and 4096 in profile['actual_native_Down_runtime_M_values'] and all(v['banks']==v['M4096_banks']==42 for v in profile['actual_attention_runtime_native_by_role'].values()) and profile['dots_Down64_captured_replay_linked_to_inherited_model_M64_execution']
 result.update(original903_r3_sources_preserved=True,only_new_precision='M64_Down_group128_RTN_W4A16_dotsaddz',original_Down64_BF16_capture_replaced_explicitly=True,actual_Down64_W4_capture_banks=42,old_Down64_BF16_capture_banks=0,dots_Down64_additional_packed_payload_bytes=profile['dots_Down64_additional_packed_payload_bytes'],dots_Down64_memory_profile_all42_accounted=True,new_W4_Python_runtime_banks=profile['dots_Down64_runtime_Python_banks'],new_W4_Python_counters_not_claimed_as_replays=True,new_W4_C64_model_graph_linked=profile['dots_Down64_captured_replay_linked_to_inherited_model_M64_execution'],previous_accuracy_not_reused=True)
 if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def audit_benchmark(*args,**kwargs):raise AssertionError('Fresh Down64 W4 Math only; no performance before C8/C64 quality')
