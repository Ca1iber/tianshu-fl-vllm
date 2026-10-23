# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
legacy=load('_dots_Down64_short_all_original',EXP/'tianshu_native_int8_prefill_service_v2_20261022/audit_profile.py');old_down=load('_dots_Down64_short_original_Down',EXP/'tianshu_native_int8_down_service_v2_20261022/audit_profile.py');new_down=load('_dots_Down64_short_explicit_Down64_replacement',EXP/'tianshu_dotsaddz_Down64_service_v2_20261023/inherited_down_audit.py');attention=load('_dots_Down64_short_original_attention',EXP/'tianshu_native_int8_attention_service_20261022/audit_profile.py');own=load('_dots_Down64_short_new_W4',EXP/'tianshu_dotsaddz_Down64_service_v2_20261023/audit_profile.py')
def audit_startup(root,state):
 r=old_down.audit_startup(root,state);r.update(attention.audit_startup(root,state))
 if state['arm']=='candidate':r.update(own.audit_startup(root,state))
 return r

def audit_runtime(root,state,official=False):
 assert official and state['mode']=='candidate' and state['arm'] in ('control','candidate') and state['workload']['input_tokens']==4096 and state['down_mode']=='candidate' and state['down_scope']=='prefill' and state['attention_mode']=='candidate';r=legacy.audit_runtime(root,state,official=True);r.update((old_down if state['arm']=='control' else new_down).audit(root,state));r.update(old_down.audit_startup(root,state));r.update(attention.audit(root,state));r.update(attention.audit_startup(root,state));assert r['actual_Down_M64_capture_banks']==0 and r['actual_native_Down_runtime_banks']==42 and 4096 in r['actual_native_Down_runtime_M_values'] and r['actual_attention_M64_capture_native_banks']==0 and all(v['banks']==v['M4096_banks']==42 for v in r['actual_attention_runtime_native_by_role'].values());assert r['actual_B64_capture_and_model_replay_verified'] and r['actual_FL_Graph_capture_and_replay_verified']
 if state['arm']=='candidate':r.update(own.audit_startup(root,state));assert r['actual_Down64_W4_capture_banks']==42 and r['old_Down64_BF16_capture_banks']==0 and r['dots_Down64_load_and_native_memory_scratch_all42_accounted'];r['new_W4_all42_capture_linked_to_actual_B64_model_replays']=True
 r.update(exact_published_r3_baseline=state['arm']=='control',original_GU_attention_KV_scheduler_sampling_LM_and_other_Down_checks_retained=True,only_changed_original_guard='candidate Down64 BF16 captures replaced with actual42 W4 captures');return r
