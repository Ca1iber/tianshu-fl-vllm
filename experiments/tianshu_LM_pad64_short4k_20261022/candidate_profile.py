# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;QUALITY=EXP/'tianshu_LM_pad64_quality_results_v2_20261022'
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
r3=load('_LM_short_original_r3',EXP/'tianshu_native_int8_attention_service_20261022/candidate_profile.py');attention=load('_LM_short_new_model',EXP/'tianshu_LM_pad64_service_20261022/candidate_profile.py');digest,read,frozen_check=r3.digest,r3.read,r3.frozen_check;CACHE_GATE=r3.parent.parent.CACHE_GATE;normalizer=load('_LM_short_original_normalizer',EXP/'tianshu_original_command_20261016/validation_profile.py');normalized_environment=normalizer.normalized_environment

def quality_gate(scope,sources=None):
 assert scope=='prefill';gate=read(QUALITY/'QUALITY_GATE_32g.json');assert gate['complete'] and gate['model_quality_qualified'] and gate['total_per_batch']==105 and gate['model_source_count']==925 and gate['controller_count']==82 and gate['C8_correct']/105>.95 and gate['C64_correct']/105>.95 and gate['new_LM_component_CUBIN_pack_memory_and_actual_calls_verified'] and gate['actual_Math_interval_separated_from_startup_probes'] and gate['C64_original_experiment_passed'] is False and gate['C64_CPU_runtime_audit_recovered']
 if sources is not None:assert sources==gate['model_frozen_sources']
 frozen_check(gate['model_frozen_sources']);frozen_check(gate['CPU_recovery_controller_sources'])
 for batch in (8,64):
  arm=gate['arms'][str(batch)];root=Path(arm['quality_root']);raw=Path(arm['raw_root']);s=read(root/'result.json');q=read(root/'accuracy_audit.json');assert s['phase']==('CPU_recovery_complete' if batch==64 else 'complete') and s['accuracy_passed'] and s['quality_batch']==batch and q['complete'] and q['total']==105 and q['correct']==arm['correct'] and q['accuracy']>.95 and s['frozen_sources']==q['frozen_sources']==gate['model_frozen_sources'];assert digest(root/'result.json')==gate['raw_files'][str(root/'result.json')] and digest(root/'accuracy_audit.json')==gate['raw_files'][str(root/'accuracy_audit.json')];frozen_check(s['frozen_sources']);frozen_check(s['controller_sources']);assert q['actual_attention_M64_capture_native_banks']==0
  if batch==64:assert read(raw/'result.json')['phase']=='audit_failed' and digest(raw/'result.json')==q['original_result_sha256'] and s['original_experiment_passed'] is False and q['CPU_coverage_reconciliation'] and all(v['banks']==v['M4096_banks']==42 for v in q['actual_attention_runtime_native_by_role'].values())
 return gate

def prepare(root,mode,scope='prefill',check_quality=True):
 assert mode in ('control','candidate') and scope=='prefill';command,environment,profile,sources=(r3 if mode=='control' else attention).prepare(root,'candidate')
 if mode=='control':published=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');assert sources==published['frozen_sources'] and command==published['server_command']
 if check_quality:quality_gate(scope,sources if mode=='candidate' else None)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_pair.sh'};paths.update(Path(p) for p in read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json')['controller_sources']);paths.update((EXP/'tianshu_LM_pad64_quality_20261022').glob('*.py'));paths.update((EXP/'tianshu_LM_pad64_audit_recovery_20261022').glob('*.py'));paths.add(EXP/'tianshu_LM_pad64_audit_recovery_20261022/RECOVERY_GATE_32g.json')
 if check_quality:paths.update({QUALITY/'QUALITY_GATE_32g.json',QUALITY/'verify_cpu.py',QUALITY/'PLAN_32g.md'})
 for module in tuple(sys.modules.values()):
  path=getattr(module,'__file__',None)
  if path and path.endswith('.py') and Path(path).is_relative_to(EXP):paths.add(Path(path).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources
