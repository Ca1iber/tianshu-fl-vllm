# SPDX-License-Identifier: Apache-2.0
import importlib.util,types
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
control=load('_LM_short_exact_r3_legacy',EXP/'tianshu_native_int8_prefill_service_v2_20261022/audit_profile.py');candidate=load('_LM_short_new_legacy',EXP/'tianshu_native_int8_prefill_service_v2_20261022/audit_profile.py');down=load('_LM_short_retained_Down',EXP/'tianshu_native_int8_down_service_v2_20261022/audit_profile.py');projection=load('_LM_short_retained_r3_QKV_O',EXP/'tianshu_native_int8_attention_service_20261022/audit_profile.py');own=load('_LM_short_actual_new_LM',EXP/'tianshu_LM_pad64_service_20261022/audit_profile.py');adapter=load('_LM_short_actual_audit_adapter',EXP/'tianshu_LM_pad64_audit_recovery_20261022/patch_runtime.py');PATCHED=adapter.install(candidate,'official')

def dense_origins(root):
 queue=[root];seen=set();found=[];target=EXP/'tianshu_nn_silu_fx_screen_20261021/dense_audit.py'
 while queue:
  m=queue.pop()
  if id(m) in seen:continue
  seen.add(id(m));p=getattr(m,'__file__',None)
  if not p or not Path(p).is_relative_to(EXP):continue
  queue.extend(v for v in vars(m).values() if isinstance(v,types.ModuleType))
  if Path(p).resolve()==target.resolve():found.append(str(Path(m.audit_runtime.__code__.co_filename).resolve()))
 return found
assert dense_origins(control) and all(p==str(EXP/'tianshu_nn_silu_fx_screen_20261021/dense_audit.py') for p in dense_origins(control));assert dense_origins(candidate) and all(p==str(EXP/'tianshu_LM_pad64_audit_recovery_20261022/official_dense_audit.py') for p in dense_origins(candidate))

def audit_runtime(root,state,official=False):
 assert official and state['mode']=='candidate' and state['arm'] in ('control','candidate') and state['workload']['input_tokens']==4096 and state['down_scope']=='prefill' and state['attention_mode']=='candidate';r=(candidate if state['arm']=='candidate' else control).audit_runtime(root,state,official=True);r.update(down.audit(root,state));r.update(down.audit_startup(root,state));r.update(projection.audit(root,state));r.update(projection.audit_startup(root,state));assert r['actual_Down_M64_capture_banks']==0 and r['actual_native_Down_runtime_banks']==42 and 4096 in r['actual_native_Down_runtime_M_values'] and r['actual_attention_M64_capture_native_banks']==0 and all(v['banks']==v['M4096_banks']==42 for v in r['actual_attention_runtime_native_by_role'].values())
 if state['arm']=='candidate':r.update(own.audit(root,state));r.update(own.audit_startup(root,state));assert r['new_LM_runtime_calls']>0 and 64 in r['new_LM_actual_runtime_M'];r['old_LM64_execution_not_claimed']=True
 r.update(exact_published_r3_baseline=(state['arm']=='control'),all_other_inherited_attention_KV_scheduler_sampling_Down_QKV_O_checks_retained=True);return r
