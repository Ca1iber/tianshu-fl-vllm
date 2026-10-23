# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;RESULTS=EXP/'tianshu_W4_LMpad_quality_results_v3_20261023';QUALITY=EXP/'tianshu_W4_LMpad_quality_v2_20261023'
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
r3=load('_LMpad_short_exact_r3',EXP/'tianshu_native_int8_attention_service_20261022/candidate_profile.py');candidate=load('_LMpad_short_new_model',EXP/'tianshu_W4_LMpad_service_v2_20261023/candidate_profile.py');digest,read,frozen_check=r3.digest,r3.read,r3.frozen_check;CACHE_GATE=r3.parent.parent.CACHE_GATE
normalizer=load('_LMpad_short_normalizer',EXP/'tianshu_original_command_20261016/validation_profile.py');normalized_environment=normalizer.normalized_environment

def quality_gate(scope,sources=None):
 assert scope=='prefill';g=read(RESULTS/'QUALITY_GATE_32g.json');assert g['complete'] and g['model_quality_qualified'] and g['total_per_batch']==105 and g['model_source_count']==944 and g['C8_correct']/105>.95 and g['C64_correct']/105>.95
 assert g['all_105_reviews_predictions_report_and_raw_SHA_verified'] and g['all_caps_in_denominator'] and g['old_LM8_BF16_and_LM64_W8_guards_explicitly_replaced'] and g['new_LM1_64_load_and_native_peak_accounting_verified'] and g['fresh_entire_batch_8_LM_precision_changed']
 assert g['verifier_source_sha256']==digest(RESULTS/'verify_cpu.py')
 if sources is not None:assert sources==g['model_frozen_sources']
 frozen_check(g['model_frozen_sources']);frozen_check(g['controller_sources']);frozen_check(g['CPU_reconciliation_sources']);assert g['C8_CPU_coverage_reconciliation'] and not g['C8_original_experiment_passed'] and not g['C8_GPU_rerun']
 for p,h in g['raw_files'].items():assert digest(p)==h
 profiles={}
 for batch in (8,64):
  root=Path(f'/workspace/benchmark-results/20261023_W4_LMpad_v2_math{batch}_tianshu');s=read(root/'result.json');quality_root=Path(g['arms'][str(batch)]['quality_root']);a=read(quality_root/'accuracy_audit.json')
  if batch==8:assert s['phase']=='audit_failed' and s['error']=='AssertionError((8, 0))' and a['CPU_coverage_reconciliation'] and not a['original_experiment_passed'] and not a['GPU_rerun'] and digest(root/'result.json')==a['original_result_sha256']
  else:assert s['phase']=='complete' and s['accuracy_passed']
  assert s['quality_batch']==batch and a['complete'] and a['total']==105 and a['correct']==g['arms'][str(batch)]['correct'] and a['accuracy']>.95 and s['frozen_sources']==a['frozen_sources']==g['model_frozen_sources'] and s['controller_sources']==g['controller_sources']
  assert a['actual_Down64_W4_capture_banks']==42 and a['old_LM1_64_BF16_and_W8_requirements_explicitly_replaced'] and a['no_original_parent_counters_modified']
  assert a['profile']['new_LM_verified_batch_W4_runtime_calls']>=10 and a['profile']['old_LM1_64_BF16_W8_calls_absent']
  profiles[batch]=normalized_environment(s['profile_environment'],root)
 assert {k for k in profiles[8] if profiles[8][k]!=profiles[64][k]}=={'TIANSHU_FUTURE_KV_MODE'} and profiles[8]['TIANSHU_FUTURE_KV_MODE']=='candidate' and profiles[64]['TIANSHU_FUTURE_KV_MODE']=='control'
 return g

def prepare(root,mode,scope='prefill',check_quality=True):
 assert mode in ('control','candidate') and scope=='prefill';command,env,profile,sources=(r3 if mode=='control' else candidate).prepare(root,'candidate')
 if mode=='control':published=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');assert sources==published['frozen_sources'] and command==published['server_command']
 if check_quality:quality_gate(scope,sources if mode=='candidate' else None)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_pair.sh'};paths.update(Path(p) for p in read(QUALITY/'CPU_GATE_32g.json')['controllers_before_CPU_gate_and_EXECUTION']);paths.update({QUALITY/'CPU_GATE_32g.json',QUALITY/'EXECUTION.sha256'})
 if check_quality:paths.update(Path(p) for p in read(RESULTS/'QUALITY_GATE_32g.json')['CPU_reconciliation_sources'])
 if check_quality:paths.update({RESULTS/'QUALITY_GATE_32g.json',RESULTS/'verify_cpu.py',RESULTS/'PLAN_32g.md',BASE/'CPU_GATE_32g.json',BASE/'EXECUTION.sha256'})
 for module in tuple(sys.modules.values()):
  p=getattr(module,'__file__',None)
  if p and p.endswith('.py') and Path(p).is_relative_to(EXP):paths.add(Path(p).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,env,profile,sources
