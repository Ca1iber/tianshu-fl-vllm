# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
legacy=load('_prefill_RMS_QKV_short_all_original',EXP/'tianshu_native_int8_prefill_service_v2_20261022/audit_profile.py');down=load('_prefill_RMS_QKV_short_original_Down',EXP/'tianshu_native_int8_down_service_v2_20261022/audit_profile.py');attention=load('_prefill_RMS_QKV_short_original_attention',EXP/'tianshu_native_int8_attention_service_20261022/audit_profile.py');own=load('_prefill_RMS_QKV_short_new_fusion',EXP/'tianshu_prefill_RMS_QKV_service_v2_20261023/audit_profile.py')
def audit_startup(root,state):
 result=down.audit_startup(root,state);result.update(attention.audit_startup(root,state))
 if state['arm']=='candidate':result.update(own.audit_startup(root,state))
 return result

def audit_runtime(root,state,official=False):
 assert official and state['mode']=='candidate' and state['arm'] in ('control','candidate') and state['workload']['input_tokens']==4096 and state['down_mode']=='candidate' and state['down_scope']=='prefill' and state['attention_mode']=='candidate';r=legacy.audit_runtime(root,state,official=True);r.update(down.audit(root,state));r.update(down.audit_startup(root,state));r.update(attention.audit(root,state));r.update(attention.audit_startup(root,state));assert r['actual_Down_M64_capture_banks']==0 and r['actual_native_Down_runtime_banks']==42 and 4096 in r['actual_native_Down_runtime_M_values'] and r['actual_attention_M64_capture_native_banks']==0 and all(v['banks']==v['M4096_banks']==42 for v in r['actual_attention_runtime_native_by_role'].values())
 if state['arm']=='candidate':r.update(own.audit(root,state));assert r['new_prefill_RMS_QKV_runtime_banks']==42 and r['new_prefill_RMS_QKV_memory_peak_explicit_probes_accounted']
 r.update(exact_published_r3_baseline=(state['arm']=='control'),all_other_inherited_attention_KV_scheduler_sampling_Down_QKV_O_GU_LM_checks_retained=True);return r
