# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
r3=load('_decode_scope_short_exact_r3',EXP/'tianshu_native_int8_attention_service_20261022/candidate_profile.py');attention=load('_decode_scope_short_new_candidate',EXP/'tianshu_QKV64_keep128_service_20261023/candidate_profile.py');digest,read,frozen_check=r3.digest,r3.read,r3.frozen_check;CACHE_GATE=r3.parent.parent.CACHE_GATE
normalizer=load('_decode_scope_short_original_normalizer',EXP/'tianshu_original_command_20261016/validation_profile.py');normalized_environment=normalizer.normalized_environment

def quality_gate(scope,sources=None):
 assert scope=='prefill';gate=read(EXP/'tianshu_QKV64_keep128_quality_results_20261023/QUALITY_GATE_32g.json');assert gate['complete'] and gate['model_quality_qualified'] and gate['total_per_batch']==105 and gate['model_source_count']==928 and gate['controller_count']==51 and gate['C8_correct']/105>.95 and gate['C64_correct']/105>.95
 assert gate['actual_new_QKV_M64_uniform_FULL_capture42_O_BF16_capture42'] and gate['nonuniform_M64_BF16_policy_verified'] and gate['QKV128_native_retained_verified']
 if sources is not None:assert sources==gate['model_frozen_sources']
 frozen_check(gate['model_frozen_sources'])
 for batch in (8,64):
  root=Path(f'/workspace/benchmark-results/20261023_QKV64_keep128_math{batch}_tianshu');s=read(root/'result.json');q=read(root/'accuracy_audit.json');assert s['phase']=='complete' and s['accuracy_passed'] and s['quality_batch']==batch and s['attention_mode']=='candidate' and s['down_scope']=='prefill' and q['complete'] and q['total']==105 and q['correct']==gate['arms'][str(batch)]['correct'] and q['accuracy']>.95 and s['frozen_sources']==q['frozen_sources']==gate['model_frozen_sources'];assert digest(root/'result.json')==gate['raw_files'][str(root/'result.json')] and digest(root/'accuracy_audit.json')==gate['raw_files'][str(root/'accuracy_audit.json')];frozen_check(s['frozen_sources']);frozen_check(s['controller_sources']);assert q['actual_attention_M64_capture_native_banks']==42
  if batch==64:assert all(v['banks']==v['M4096_banks']==42 for v in q['actual_attention_runtime_native_by_role'].values())
 return gate

def prepare(root,mode,scope='prefill',check_quality=True):
 assert mode in ('control','candidate') and scope=='prefill';command,environment,profile,sources=(r3 if mode=='control' else attention).prepare(root,'candidate')
 if mode=='control':published=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');assert sources==published['frozen_sources'] and command==published['server_command']
 if check_quality:quality_gate(scope,sources if mode=='candidate' else None)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_pair.sh'};paths.update(Path(p) for p in read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json')['controller_sources']);paths.update((EXP/'tianshu_QKV64_keep128_quality_20261023').glob('*.py'))
 if check_quality:paths.update({EXP/'tianshu_QKV64_keep128_quality_results_20261023/QUALITY_GATE_32g.json',EXP/'tianshu_QKV64_keep128_quality_results_20261023/verify_cpu.py',EXP/'tianshu_QKV64_keep128_quality_results_20261023/PLAN_32g.md'})
 for m in tuple(sys.modules.values()):
  path=getattr(m,'__file__',None)
  if path and path.endswith('.py') and Path(path).is_relative_to(EXP):paths.add(Path(path).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources
