# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SERVICE=EXP/'tianshu_RMS_A8_GU_service_v2_20261023';HOLDOUT=EXP/'tianshu_native_int8_attention42_20261022'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
parent=load('_RMS_GU_service',SERVICE/'candidate_profile.py');legacy=load('_RMS_GU_all_r3_quality_audits',EXP/'tianshu_native_int8_attention_quality_20261022/quality_profile.py');own=load('_RMS_GU_own_audit',SERVICE/'audit_profile.py');digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check;BENCH_SHA,csv_rows,normalized_environment=legacy.BENCH_SHA,legacy.csv_rows,legacy.normalized_environment

def prepare(root,screen_root,batch=8,scope='prefill'):
 assert screen_root.resolve()==HOLDOUT.resolve() and batch in (8,64) and scope=='prefill';command,environment,profile,sources=parent.prepare(root,'candidate')
 if batch==64:assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate';profile['TIANSHU_FUTURE_KV_MODE']='control';environment.update(profile)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_quality.sh'};release=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');paths.update(Path(p) for p in release['controller_sources'])
 for m in tuple(sys.modules.values()):
  p=getattr(m,'__file__',None)
  if p and p.endswith('.py') and Path(p).is_relative_to(EXP):paths.add(Path(p).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  t=root/'controller_source_snapshot'/Path(p).relative_to('/');t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(Path(p).read_bytes());assert digest(t)==h
 return command,environment,profile,sources

def audit_startup(root,state):
 result=legacy.audit_startup(root,state);result.update(own.audit(root,state));return result

def audit_runtime(root,state,official=False):
 assert not official;result=legacy.audit_runtime(root,state,official=False);result.update(own.audit(root,state));return result

def audit_math(root,state,profile,write=True):
 result=legacy.audit_math(root,state,profile,write=False);assert profile['new_RMS_A8_GU_all42_static_M64_capture_and_real_binaries_verified'] and profile['new_RMS_A8_GU_no_fictitious_old_quantizer_calls'] and profile['new_RMS_A8_GU_dynamic_ranges_unchanged'];result.update(RMS_A8_GU_new_model=True,RMS_A8_GU_static_M64_only=True,old_GU_GEMM_coverage_is_actual_kernel_index_not_old_A8_calls=True)
 if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def audit_benchmark(*args,**kwargs):raise AssertionError('Math-only candidate; future performance requires a separate frozen controller')
