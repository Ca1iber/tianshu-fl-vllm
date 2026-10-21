# SPDX-License-Identifier: Apache-2.0
"""Actual C64 Math for the new BM8 path; admission switch only."""
import json
from pathlib import Path
from framework_profile import BASE,SCREEN,BENCH_SHA,load,read,digest,frozen_check,csv_rows,normalized_environment,original_math,controllers,MATH8
import framework_profile as canonical
OLD=load('_BM8497_C64_existing_checks',BASE.parent/'tianshu_nn_silu_fx_c64_quality_20261021/auxiliary_profile.py')
quality=load('_BM8497_actual_C64_quality',BASE/'quality_runtime.py');quality.parent=OLD.dense

def prepare_runtime(root,screen_root,require_screen=True):
 command,environment,profile,sources=canonical.prepare_runtime(root,screen_root,require_screen)
 assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate';profile['TIANSHU_FUTURE_KV_MODE']='control';environment.update(profile)
 return command,environment,profile,sources

def prepare(root,screen_root):
 pre=read(BASE/'CPU_GATE_32g.json');assert pre['complete'];frozen_check(pre['sources'])
 official=read(MATH8/'result.json');accuracy=read(MATH8/'accuracy_audit.json')
 assert official['phase']=='complete' and accuracy['complete'] and accuracy['total']==105 and accuracy['accuracy']>.95
 answer=prepare_runtime(root,screen_root);command,environment,profile,sources=answer
 assert sources==official['frozen_sources'] and command==official['server_command'];controllers(root);return answer

def audit_runtime(root,state,official=False):
 assert not official and state['profile_environment']['TIANSHU_FUTURE_KV_MODE']=='control'
 result=quality.audit_runtime(root,state,official=False);result.update(canonical.extra.audit(root,state))
 result=canonical.full.audit_extension(root,state,result,require_C64=True)
 result.update(actual_C8_Math_only_not_C64_request_claim=False,actual_C64_fused_replay_verified=True,auxiliary_Math_batch=64,official_Math8_not_replaced=True,numerical_model_source_count=497)
 return result

def audit_math(root,state,profile,write=True):
 result=original_math(root,state,profile,write=False)
 assert profile['actual_C64_fused_replay_verified'] and profile['verified_dense_batch']==64 and profile['LM_head_precision_in_verified_batch']=='INT8'
 assert profile['actual_range_fused_all_42_layers_verified'] and profile['actual_range_fused_M4096_all42_verified'] and profile['BM8_actual_C64_request_graph_replay_verified']
 official=read(MATH8/'result.json');assert state['frozen_sources']==official['frozen_sources'] and len(state['frozen_sources'])==497
 actual=normalized_environment(state['profile_environment'],root);expected=normalized_environment(official['profile_environment'],MATH8)
 assert {k for k in actual if actual[k]!=expected[k]}=={'TIANSHU_FUTURE_KV_MODE'}
 assert actual['TIANSHU_FUTURE_KV_MODE']=='control' and expected['TIANSHU_FUTURE_KV_MODE']=='candidate' and state['server_command']==official['server_command']
 result.update(fresh_math_for_original_command_profile=False,supplementary_C64_quality=True,official_Math8_not_replaced=True,actual_C64_fused_replay_verified=True,actual_C64_LM_INT8_exercised=True,actual_prefill_range_fused_all_42_layers_verified=True,actual_M4096_range_fused_all42_verified=True,BM8_actual_C64_replay_verified=True,model_frozen_sources=state['frozen_sources'],only_admission_environment_difference=True,controller_sources=state['controller_sources'],model_quality_bound_to_same497_source_map=True)
 if write:
  (root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n');archive={k:v for k,v in result.items() if k not in ('profile','frozen_sources','controller_sources')};archive.update(raw_root=str(root),verifier_source=str(BASE/'auxiliary_profile.py'),verifier_sha256=digest(__file__))
  (BASE/'C64_QUALITY_GATE_32g.json').write_text(json.dumps(archive,indent=2)+'\n')
 return result

def audit_benchmark(*args,**kwargs):raise AssertionError('Supplementary C64 quality is not an official benchmark')
