# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
legacy=load('_attention_short_all_r1_runtime',EXP/'tianshu_native_int8_prefill_service_v2_20261022/audit_profile.py');down=load('_attention_short_retained_Down',EXP/'tianshu_native_int8_down_service_v2_20261022/audit_profile.py');attention=load('_attention_short_own_projection_audit',EXP/'tianshu_native_int8_attention_service_20261022/audit_profile.py')
def audit_runtime(root,state,official=False):
 assert official and state['mode']=='candidate' and state['workload']['input_tokens']==4096 and state['down_mode']=='candidate' and state['down_scope']=='prefill';result=legacy.audit_runtime(root,state,official=True);result.update(down.audit(root,state));result.update(down.audit_startup(root,state));assert result['actual_native_Down_runtime_banks']==42 and 4096 in result['actual_native_Down_runtime_M_values'] and result['actual_Down_M64_capture_banks']==0
 if state['arm']=='candidate':
  assert state['attention_mode']=='candidate';result.update(attention.audit(root,state));result.update(attention.audit_startup(root,state));assert all(v['banks']==v['M4096_banks']==42 for v in result['actual_attention_runtime_native_by_role'].values()) and result['actual_attention_M64_capture_native_banks']==0
 else:assert state['arm']=='control' and not (root/'native_attention_records').exists()
 result.update(exact_published_r2_baseline=(state['arm']=='control'),all_inherited_attention_KV_scheduler_sampling_and_Down_checks_retained=True);return result
