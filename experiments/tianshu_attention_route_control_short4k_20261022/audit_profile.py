# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
legacy=load('_route_control_inherited_runtime',EXP/'tianshu_native_int8_prefill_service_v2_20261022/audit_profile.py');down=load('_route_control_retained_Down',EXP/'tianshu_native_int8_down_service_v2_20261022/audit_profile.py');attention=load('_route_control_actual_BF16',EXP/'tianshu_native_int8_attention_service_20261022/audit_profile.py')
def audit_runtime(root,state,official=False):
 assert official and state['mode']=='candidate' and state['arm']=='control' and state['attention_mode']=='control' and state['down_mode']=='candidate' and state['down_scope']=='prefill';result=legacy.audit_runtime(root,state,official=True);result.update(down.audit(root,state));result.update(down.audit_startup(root,state));result.update(attention.audit(root,state));result.update(attention.audit_startup(root,state));assert result['actual_native_Down_runtime_banks']==42 and 4096 in result['actual_native_Down_runtime_M_values'] and all(v['banks']==0 for v in result['actual_attention_runtime_native_by_role'].values());result.update(same_hook_BF16_control=True,new_attention_runtime_INT8_absent=True,quality_not_independently_qualified=True,all_original_runtime_checks_retained=True);return result
