# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SERVICE=EXP/'tianshu_LM_pad64_service_20261022';HOLDOUT=EXP/'tianshu_LM_pad64_20261022'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
parent=load('_LM_pad64_live_profile',SERVICE/'candidate_profile.py');legacy=load('_LM_pad64_retained_full_r3',EXP/'tianshu_native_int8_attention_quality_20261022/quality_profile.py');own=load('_LM_pad64_new_explicit_audit',SERVICE/'audit_profile.py');PATCHED=own.patch_dense_modules(legacy);digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check;BENCH_SHA,csv_rows,normalized_environment=legacy.BENCH_SHA,legacy.csv_rows,legacy.normalized_environment

def prepare(root,screen_root,batch=8,scope='prefill'):
 assert screen_root.resolve()==HOLDOUT.resolve() and batch in (8,64) and scope=='prefill';g=read(HOLDOUT/'INDEPENDENT_CPU_GATE_32g.json');assert g['complete'] and g['dynamic_range_lead'] and g['raw_result_sha256']==digest('/workspace/benchmark-results/20261022_LM_pad64_GPU_tianshu/result.json');command,environment,profile,sources=parent.prepare(root,'candidate')
 if batch==64:assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate';profile['TIANSHU_FUTURE_KV_MODE']='control';environment.update(profile)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_quality.sh',BASE/'LM_COMPONENT_CACHE.sha256'};release=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');paths.update(Path(p) for p in release['controller_sources'])
 for module in tuple(sys.modules.values()):
  path=getattr(module,'__file__',None)
  if path and path.endswith('.py') and Path(path).is_relative_to(EXP):paths.add(Path(path).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources

def audit_startup(root,state):
 r=legacy.audit_startup(root,state);r.update(own.audit_startup(root,state));return r

def audit_runtime(root,state,official=False):
 assert not official and state['attention_mode']=='candidate' and state['down_scope']=='prefill';r=legacy.audit_runtime(root,state,official=False);r.update(own.audit(root,state));r.update(audit_startup(root,state));r['explicit_replaced_legacy_dense_audit_copies']=PATCHED;r['LM_precision_and_route_unchanged']=False;r['old_LM64_execution_not_claimed']=True;return r

def audit_math(root,state,profile,write=True):
 r=legacy.audit_math(root,state,profile,write=False);assert profile['new_LM_CUBIN_exact'] and profile['new_LM_original_pack_same'] and profile['new_LM_profile_peak_accounted'] and profile['legacy_LM64_guard_explicitly_replaced_by_new_integer_counter'];r.update(new_LM_actual_runtime_M=profile['new_LM_actual_runtime_M'],new_LM_runtime_calls=profile['new_LM_runtime_calls'],new_LM64_original_bits_exact=True,new_LM_smaller_range_precision_requires_this_fresh_quality=True,new_LM_minimum_M=read(SERVICE/'selected_plan.json')['minimum_M'],LM_precision_and_route_unchanged=False,old_LM64_execution_not_claimed=True,original_r3_QKV_O_Down_GU_KV_sampling_runtime_checks_retained=True,explicit_LM64_audit_replacement=True)
 if state['quality_batch']==64:assert profile['new_LM_runtime_calls']>0 and 64 in profile['new_LM_actual_runtime_M']
 if write:(root/'accuracy_audit.json').write_text(json.dumps(r,indent=2)+'\n')
 return r

def audit_benchmark(*args,**kwargs):raise AssertionError('Math-only LM range controller; separate frozen short/official needed')
