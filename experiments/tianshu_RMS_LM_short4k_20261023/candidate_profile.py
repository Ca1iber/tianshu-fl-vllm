# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
r3=load('_RMS_LM_short_exact_r3',EXP/'tianshu_native_int8_attention_service_20261022/candidate_profile.py');combined=load('_RMS_LM_short_combined',EXP/'tianshu_RMS_LM_service_20261023/candidate_profile.py');digest,read,frozen_check=r3.digest,r3.read,r3.frozen_check;CACHE_GATE=r3.parent.parent.CACHE_GATE
normalizer=load('_RMS_LM_short_normalizer',EXP/'tianshu_original_command_20261016/validation_profile.py');normalized_environment=normalizer.normalized_environment

def quality_gate(scope,sources=None):
 assert scope=='prefill';gate=read(EXP/'tianshu_RMS_LM_quality_results_20261023/QUALITY_GATE_32g.json');assert gate['complete'] and gate['model_quality_qualified'] and gate['total_per_batch']==105 and gate['model_source_count']==944 and gate['controller_count']==91 and gate['C8_correct']/105>.95 and gate['C64_correct']/105>.95
 assert gate['new_combined_RMS_LM_model'] and gate['both_parent_profiles_exact_subsets'] and gate['old_LM64_execution_not_claimed'] and gate['new_RMS_A8_GU_all42_capture_and_real_binaries_verified'] and gate['no_fictitious_old_quantizer_calls'] and gate['all_dynamic_ranges_unchanged'] and gate['all_inherited_and_new_runtime_audits_independently_recomputed']
 if sources is not None:assert sources==gate['model_frozen_sources']
 frozen_check(gate['model_frozen_sources'])
 for batch in (8,64):
  root=Path(f'/workspace/benchmark-results/20261023_RMS_LM_math{batch}_tianshu');s=read(root/'result.json');q=read(root/'accuracy_audit.json');assert s['phase']=='complete' and s['accuracy_passed'] and s['quality_batch']==batch and s['attention_mode']=='candidate' and s['down_scope']=='prefill' and q['complete'] and q['total']==105 and q['correct']==gate['arms'][str(batch)]['correct'] and q['accuracy']>.95 and s['frozen_sources']==q['frozen_sources']==gate['model_frozen_sources'];assert digest(root/'result.json')==gate['raw_files'][str(root/'result.json')] and digest(root/'accuracy_audit.json')==gate['raw_files'][str(root/'accuracy_audit.json')];frozen_check(s['frozen_sources']);frozen_check(s['controller_sources']);assert q['actual_attention_M64_capture_native_banks']==0 and q['RMS_A8_GU_new_model'] and q['new_combined_RMS_LM_model'] and q['old_LM64_execution_not_claimed']
  if batch==64:assert all(v['banks']==v['M4096_banks']==42 for v in q['actual_attention_runtime_native_by_role'].values()) and q['new_LM_runtime_calls']>0 and 64 in q['new_LM_actual_runtime_M']
 return gate

def prepare(root,mode,scope='prefill',check_quality=True):
 assert mode in ('control','candidate') and scope=='prefill';command,environment,profile,sources=(r3 if mode=='control' else combined).prepare(root,'candidate')
 if mode=='control':published=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');assert sources==published['frozen_sources'] and command==published['server_command']
 if check_quality:quality_gate(scope,sources if mode=='candidate' else None)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_pair.sh'};paths.update(Path(p) for p in read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json')['controller_sources']);paths.update((EXP/'tianshu_RMS_LM_quality_20261023').glob('*.py'))
 if check_quality:paths.update({EXP/'tianshu_RMS_LM_quality_results_20261023/QUALITY_GATE_32g.json',EXP/'tianshu_RMS_LM_quality_results_20261023/verify_cpu.py',EXP/'tianshu_RMS_LM_quality_results_20261023/PLAN_32g.md'})
 for m in tuple(sys.modules.values()):
  path=getattr(m,'__file__',None)
  if path and path.endswith('.py') and Path(path).is_relative_to(EXP):paths.add(Path(path).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources
