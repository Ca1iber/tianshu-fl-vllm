# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
legacy=load('_down_short_all_r1_runtime_checks',EXP/'tianshu_native_int8_prefill_service_v2_20261022/audit_profile.py');down=load('_down_short_actual_native_Down_checks',EXP/'tianshu_native_int8_down_service_v2_20261022/audit_profile.py')
def audit_runtime(root,state,official=False):
 assert official and state['mode']=='candidate' and state['workload']['input_tokens']==4096
 result=legacy.audit_runtime(root,state,official=True)
 if state['arm']=='candidate':
  result.update(down.audit(root,state));assert result['actual_native_Down_runtime_banks']==42 and 4096 in result['actual_native_Down_runtime_M_values'] and result['actual_Down_M64_capture_banks']==(42 if state['down_scope']=='all' else 0)
  result.update(down.audit_startup(root,state))
 else:assert state['arm']=='control' and not (root/'native_down_records').exists()
 result.update(exact_published_r1_baseline=(state['arm']=='control'),all_inherited_attention_KV_scheduler_sampling_checks_retained=True)
 return result
