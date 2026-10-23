# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SERVICE=EXP/'tianshu_prefill_RMS_QKV_service_20261023';HOLDOUT=EXP/'tianshu_native_int8_attention42_20261022'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
parent=load('_prefill_RMS_QKV_quality_service',SERVICE/'candidate_profile.py');legacy=load('_prefill_RMS_QKV_all_original_r3_audits',EXP/'tianshu_native_int8_attention_quality_20261022/quality_profile.py');own=load('_prefill_RMS_QKV_quality_new_audit',SERVICE/'audit_profile.py');digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
BENCH_SHA,csv_rows,normalized_environment=legacy.BENCH_SHA,legacy.csv_rows,legacy.normalized_environment

def prepare(root,screen_root,batch=8,scope='prefill'):
 assert screen_root.resolve()==HOLDOUT.resolve() and batch in (8,64) and scope=='prefill';gate=read(SERVICE/'CPU_GATE_32g.json');assert gate['complete'];command,environment,profile,sources=parent.prepare(root,'candidate');assert len(sources)==gate['expected_model_source_count']
 if batch==64:assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate';profile['TIANSHU_FUTURE_KV_MODE']='control';environment.update(profile)
 paths={EXP/'tianshu_prefill_RMS_QKV_quality_results_20261023/verify_cpu.py',EXP/'tianshu_prefill_RMS_QKV_quality_results_20261023/PLAN_32g.md'}|set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_quality.sh'};release=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');paths.update(Path(p) for p in release['controller_sources']);paths.update((EXP/'tianshu_native_int8_attention_quality_20261022').glob('*.py'))
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
 assert not official and state['attention_mode']=='candidate' and state['down_scope']=='prefill';result=legacy.audit_runtime(root,state,official=False);result.update(own.audit(root,state));result.update(audit_startup(root,state));result['all_original_r3_runtime_checks_retained']=True;return result

def audit_math(root,state,profile,write=True):
 result=legacy.audit_math(root,state,profile,write=False);assert profile['all_original_r3_runtime_checks_retained'] and profile['new_prefill_RMS_QKV_startup_all42_static4096_verified'] and profile['new_prefill_RMS_QKV_memory_peak_explicit_probes_accounted'] and profile['new_prefill_RMS_QKV_general_and_M64_graphs_unchanged'] and profile['new_prefill_RMS_QKV_old_custom_op_and_standalone_A8_calls_not_claimed'] and profile['new_prefill_RMS_QKV_same_native_MM_epilogue_and_original_packs_verified']
 if state['quality_batch']==64:assert profile['new_prefill_RMS_QKV_runtime_banks']==42
 result.update(new_prefill_RMS_QKV_model=True,new_prefill_RMS_QKV_actual_compile_sizes=profile['new_prefill_RMS_QKV_actual_compile_sizes'],new_prefill_RMS_QKV_runtime_banks=profile['new_prefill_RMS_QKV_runtime_banks'],new_prefill_RMS_QKV_runtime_bank_ids=profile['new_prefill_RMS_QKV_runtime_bank_ids'],original903_r3_sources_retained=True,actual_C8_unhit_new_paths_not_claimed_as_C64=True)
 if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def audit_benchmark(*args,**kwargs):raise AssertionError('Fresh Math-only prefill fusion controller')
