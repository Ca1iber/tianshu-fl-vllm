# SPDX-License-Identifier: Apache-2.0
"""Separate exact-r3 and explicitly adapted LM64-W4 audit instances."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
control=load('_LM64_short_control_all_guards',EXP/'tianshu_dots_Down64_short4k_v2_20261023/audit_profile.py');candidate=load('_LM64_short_candidate_all_guards',EXP/'tianshu_dots_Down64_short4k_v2_20261023/audit_profile.py');dense=load('_LM64_short_explicit_LM64_guard_replacement',EXP/'tianshu_dotsaddz_LM64_service_20261023/inherited_full_dense_audit.py');own=load('_LM64_short_actual_LM64_W4',EXP/'tianshu_dotsaddz_LM64_service_20261023/audit_profile.py')
assert control.legacy.legacy.lower.parent is not candidate.legacy.legacy.lower.parent
candidate.legacy.legacy.lower.parent=dense;candidate.legacy.legacy.dense=dense

def audit_startup(root,state):
 r=(control if state['arm']=='control' else candidate).audit_startup(root,state)
 if state['arm']=='candidate':r.update(own.audit_startup(root,state))
 return r

def audit_runtime(root,state,official=False):
 assert official;r=(control if state['arm']=='control' else candidate).audit_runtime(root,state,official=True)
 if state['arm']=='candidate':
  r.update(own.audit_runtime(root,state,minimum_calls=100));assert r['new_W4_all42_capture_linked_to_actual_B64_model_replays'] and r['new_LM64_W4_actual_runtime_calls']>=100 and r['old_LM64_W8_calls_absent'];r.update(original_GU_attention_KV_scheduler_sampling_LM_and_other_Down_checks_retained=False,original_GU_attention_KV_scheduler_sampling_other_Down_and_LM8_checks_retained=True,original_LM64_INT8_precision_retained=False,LM64_requirement_explicitly_replaced_with_actual_native_W4_calls=True,only_changed_original_guards='Down64 BF16 capture->actual42 W4 capture and LM64 W8 calls->actual native W4 calls')
 return r
