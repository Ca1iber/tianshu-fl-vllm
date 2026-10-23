# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;RESULTS=EXP/'tianshu_dots_Down64_quality_results_v2_20261023'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
r3=load('_dots_Down64_short_exact_r3',EXP/'tianshu_native_int8_attention_service_20261022/candidate_profile.py');candidate=load('_dots_Down64_short_new_service',EXP/'tianshu_dotsaddz_Down64_service_v2_20261023/candidate_profile.py');digest,read,frozen_check=r3.digest,r3.read,r3.frozen_check;CACHE_GATE=r3.parent.parent.CACHE_GATE
normalizer=load('_dots_Down64_short_normalizer',EXP/'tianshu_original_command_20261016/validation_profile.py');normalized_environment=normalizer.normalized_environment

def quality_gate(scope,sources=None):
 assert scope=='prefill';g=read(RESULTS/'QUALITY_GATE_32g.json');assert g['complete'] and g['model_quality_qualified'] and g['total_per_batch']==105 and g['model_source_count']==916 and g['controller_count']==85 and g['C8_correct']/105>.95 and g['C64_correct']/105>.95;assert g['new_precision_only_M64_Down_RTNg128_W4'] and g['new_all42_W4_captures_and_old_BF16_capture0_verified'] and g['new_peak_scratch_accounted_inside_actual_native_memory_profile'] and g['actual_parent_op_schema_bank_and_other_M_delegation_verified'] and g['full_profile_and_accuracy_audit_recomputed_exact'] and g['all_inherited_and_new_runtime_audits_independently_recomputed']
 if sources is not None:assert sources==g['model_frozen_sources']
 frozen_check(g['model_frozen_sources']);frozen_check(g['controller_sources'])
 for batch in (8,64):
  root=Path(f'/workspace/benchmark-results/20261023_dots_Down64_v2_math{batch}_tianshu');s=read(root/'result.json');q=read(root/'accuracy_audit.json');assert s['phase']=='complete' and s['accuracy_passed'] and s['quality_batch']==batch and q['complete'] and q['total']==105 and q['correct']==g['arms'][str(batch)]['correct'] and q['accuracy']>.95 and s['frozen_sources']==q['frozen_sources']==g['model_frozen_sources'];assert digest(root/'result.json')==g['raw_files'][str(root/'result.json')] and digest(root/'accuracy_audit.json')==g['raw_files'][str(root/'accuracy_audit.json')];assert q['actual_Down64_W4_capture_banks']==42 and q['old_Down64_BF16_capture_banks']==0 and q['only_new_precision']=='M64_Down_group128_RTN_W4A16_dotsaddz'
  if batch==64:assert q['new_W4_C64_model_graph_linked']
 return g

def prepare(root,mode,scope='prefill',check_quality=True):
 assert mode in ('control','candidate') and scope=='prefill';command,env,profile,sources=(r3 if mode=='control' else candidate).prepare(root,'candidate')
 if mode=='control':published=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');assert sources==published['frozen_sources'] and command==published['server_command']
 if check_quality:quality_gate(scope,sources if mode=='candidate' else None)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_pair.sh'};paths.update(Path(p) for p in read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json')['controller_sources']);paths.update((EXP/'tianshu_dotsaddz_Down64_quality_v2_20261023').glob('*.py'))
 if check_quality:paths.update({RESULTS/'QUALITY_GATE_32g.json',RESULTS/'verify_cpu.py',RESULTS/'PLAN_32g.md'})
 for m in tuple(sys.modules.values()):
  p=getattr(m,'__file__',None)
  if p and p.endswith('.py') and Path(p).is_relative_to(EXP):paths.add(Path(p).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,env,profile,sources
