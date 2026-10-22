# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
SERVICE=EXP/'tianshu_native_int8_down_service_v2_20261022';HOLDOUT=EXP/'tianshu_native_int8_down42_20261022'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
parent=load('_down_quality_service_profile',SERVICE/'candidate_profile.py');legacy=load('_down_quality_inherited_complete_audits',EXP/'tianshu_native_int8_prefill_validation_v2_20261022/quality_profile.py');down=load('_down_quality_own_audit',SERVICE/'audit_profile.py')
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
BENCH_SHA,csv_rows,normalized_environment=legacy.BENCH_SHA,legacy.csv_rows,legacy.normalized_environment
audit_startup=down.audit_startup

def prepare(root,screen_root,batch=8,scope='all'):
 assert screen_root.resolve()==HOLDOUT.resolve() and batch in (8,64) and scope in ('all','prefill')
 gate=read(HOLDOUT/'INDEPENDENT_CPU_GATE_32g.json');assert gate['complete'] and gate['raw_result_sha256']==digest(Path(gate['raw_directory'])/'result.json')
 command,environment,profile,sources=parent.prepare(root,'candidate',scope)
 if batch==64:assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate';profile['TIANSHU_FUTURE_KV_MODE']='control';environment.update(profile)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_quality.sh'}
 release=read(EXP/'tianshu_native_int8_prefill_release_v2_20261022/serve_profile_template.json');paths.update(Path(p) for p in release['controller_sources']);paths.update((EXP/'tianshu_native_int8_prefill_validation_v2_20261022').glob('*.py'))
 for module in tuple(sys.modules.values()):
  path=getattr(module,'__file__',None)
  if path and path.endswith('.py') and Path(path).is_relative_to(EXP):paths.add(Path(path).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources

def audit_runtime(root,state,official=False):
 assert not official and state['mode']=='candidate' and state['down_mode']=='candidate'
 result=legacy.audit_runtime(root,state,official=False);result.update(down.audit(root,state));result.update(all_original_runtime_checks_retained=True,actual_new_Down_accuracy_required=True)
 result.update(down.audit_startup(root,state))
 return result

def audit_math(root,state,profile,write=True):
 result=legacy.audit_math(root,state,profile,write=False);batch=state['quality_batch']
 assert profile['all_original_runtime_checks_retained'] and profile['new_Down_memory_profile_all42_accounted']
 if batch==64:assert profile['actual_native_Down_runtime_banks']==42 and 4096 in profile['actual_native_Down_runtime_M_values'] and profile['actual_Down_M64_capture_banks']==(42 if state['down_scope']=='all' else 0)
 result.update(down_scope=state['down_scope'],down_mode=state['down_mode'],actual_native_Down_runtime_banks=profile['actual_native_Down_runtime_banks'],actual_native_Down_runtime_M_values=profile['actual_native_Down_runtime_M_values'],actual_Down_M64_capture_banks=profile['actual_Down_M64_capture_banks'],C8_unhit_path_not_claimed=True,previous_accuracy_not_reused=True,model_original877_sources_retained=True)
 if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def audit_benchmark(*args,**kwargs):raise AssertionError('Math-only controller: new short4K and official4K require separate frozen controllers')
