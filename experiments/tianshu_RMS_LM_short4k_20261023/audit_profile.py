# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
legacy=load('_RMS_LM_short_retained_all_and_real_LM',EXP/'tianshu_LM_pad64_short4k_20261023/audit_profile.py');rms=load('_RMS_LM_short_actual_norm',EXP/'tianshu_RMS_A8_GU_service_v2_20261023/audit_profile.py')
def audit_runtime(root,state,official=False):
 assert official;result=legacy.audit_runtime(root,state,official=True)
 if state['arm']=='candidate':
  result.update(rms.audit(root,state));assert result['new_LM_runtime_calls']>0 and 64 in result['new_LM_actual_runtime_M'] and result['new_RMS_A8_GU_all42_static_M64_capture_and_real_binaries_verified'] and result['new_RMS_A8_GU_no_fictitious_old_quantizer_calls'];result.update(new_combined_RMS_LM_model=True,LM_precision_and_route_unchanged=False,old_LM64_execution_not_claimed=True,RMS_record_LM_unchanged_flag_scope='RMS hook only; actual combined LM audited separately')
 return result
