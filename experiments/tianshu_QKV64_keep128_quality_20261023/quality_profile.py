# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SERVICE=EXP/'tianshu_QKV64_keep128_service_20261023';HOLDOUT=EXP/'tianshu_native_int8_attention64_20261022'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
parent=load('_attention_decode64_quality_service',SERVICE/'candidate_profile.py');legacy=load('_attention_decode64_quality_all_r2_audits',EXP/'tianshu_native_int8_down_quality_v2_20261022/quality_profile.py');own=load('_attention_decode64_quality_new_audit',SERVICE/'audit_profile.py');digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
BENCH_SHA,csv_rows,normalized_environment=legacy.BENCH_SHA,legacy.csv_rows,legacy.normalized_environment

def prepare(root,screen_root,batch=8,scope='prefill'):
 assert screen_root.resolve()==HOLDOUT.resolve() and batch in (8,64) and scope=='prefill';gate=read(HOLDOUT/'INDEPENDENT_CPU_GATE_32g.json');assert gate['complete'] and gate['raw_result_sha256']==digest(Path(gate['raw_directory'])/'result.json');command,environment,profile,sources=parent.prepare(root,'candidate')
 if batch==64:assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate';profile['TIANSHU_FUTURE_KV_MODE']='control';environment.update(profile)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_quality.sh'};release=read(EXP/'tianshu_native_int8_down_prefill_release_v2_20261022/serve_profile_template.json');paths.update(Path(p) for p in release['controller_sources']);paths.update((EXP/'tianshu_native_int8_down_quality_v2_20261022').glob('*.py'))
 for module in tuple(sys.modules.values()):
  path=getattr(module,'__file__',None)
  if path and path.endswith('.py') and Path(path).is_relative_to(EXP):paths.add(Path(path).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources

def audit_startup(root,state):
 result=legacy.audit_startup(root,state);result.update(own.audit_startup(root,state));return result

def audit_runtime(root,state,official=False):
 assert not official and state['attention_mode']=='candidate' and state['down_scope']=='prefill';result=legacy.audit_runtime(root,state,official=False);result.update(own.audit(root,state));result.update(audit_startup(root,state));result['all_original_r2_runtime_checks_retained']=True;return result

def audit_math(root,state,profile,write=True):
 result=legacy.audit_math(root,state,profile,write=False);assert profile['all_original_r2_runtime_checks_retained'] and profile['new_attention_memory_profile_all84_accounted'] and profile['actual_attention_M64_capture_native_banks']==42
 if state['quality_batch']==64:assert all(row['banks']==row['M4096_banks']==42 for row in profile['actual_attention_runtime_native_by_role'].values())
 result.update(attention_mode=state['attention_mode'],attention_scope='FULL_uniform_QKV64_and_QKV128_512_4096_O512_4096',actual_attention_runtime_native_by_role=profile['actual_attention_runtime_native_by_role'],actual_attention_M64_capture_native_banks=profile['actual_attention_M64_capture_native_banks'],new_attention_packed_payload_bytes=profile['new_attention_packed_payload_bytes'],model_original915_sources_retained=True,actual_C8_unhit_paths_not_claimed_as_C64=True)
 if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def audit_benchmark(*args,**kwargs):raise AssertionError('Math-only QKV/O controller; separate frozen performance controllers required')
