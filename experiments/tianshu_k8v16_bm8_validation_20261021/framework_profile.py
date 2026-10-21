# SPDX-License-Identifier: Apache-2.0
"""Fresh quality for the exact BM8 NN497 runtime source map."""
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SERVICE=EXP/'tianshu_k8v16_bm8_service_20261021';SCREEN=EXP/'tianshu_k8v16_bm8_service_results_20261021'
MATH8=Path('/workspace/benchmark-results/20261021_BM8_math8_tianshu')
sys.path.insert(0,str(EXP/'tianshu_original_command_20261016'))
from validation_profile import BENCH_SHA,audit_math as original_math,audit_benchmark as original_benchmark,csv_rows,normalized_environment,official_command

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
parent=load('_BM8497_quality_profile',SERVICE/'candidate_profile.py');digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
quality=load('_BM8497_C8_retained_quality',BASE/'quality_runtime.py');extra=load('_BM8497_range_quality',BASE/'range_runtime_audit.py');full=load('_BM8497_full_service_audit',SERVICE/'audit_profile.py')

def prepare_runtime(root,screen_root,require_screen=True):
 assert screen_root.resolve()==SCREEN.resolve()
 command,environment,profile,sources=parent.prepare(root,'candidate')
 assert len(sources)==497 and len(profile)==62 and command==official_command()
 if require_screen:
  gate=read(SCREEN/'SCREEN_GATE_32g.json');assert gate['complete'] and gate['candidate_passes_perf_screen'] and gate['throughput_improvement_percent']>.5 and gate['all_spreads_below1_percent']
  candidate=read(Path(gate['arms']['candidate']['raw_root'])/'result.json');assert candidate['phase']=='complete'
  assert sources==candidate['frozen_sources'] and command==candidate['server_command']
  assert normalized_environment(profile,root)==normalized_environment(candidate['profile_environment'],candidate['output_root'])
 return command,environment,profile,sources

def controllers(root):
 paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'CPU_GATE_32g.json',BASE/'PLAN_32g.md',SCREEN/'SCREEN_GATE_32g.json',SCREEN/'KEY_ROUNDS_32g.csv',SCREEN/'verify_pair_cpu.py',EXP/'tianshu_original_command_20261016/validation_profile.py']
 sources={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
 for p,h in sources.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return sources

def prepare(root,screen_root):
 pre=read(BASE/'CPU_GATE_32g.json');assert pre['complete'] and pre['runtime_source_count']==497;frozen_check(pre['sources'])
 answer=prepare_runtime(root,screen_root);controllers(root);return answer

def audit_runtime(root,state,official=False):
 if official:return full.audit_runtime(root,state,official=True)
 result=quality.audit_runtime(root,state,official=False);result.update(extra.audit(root,state,require_fused=False))
 return full.audit_extension(root,state,result,require_C64=False)

def audit_math(root,state,profile,write=True):
 result=original_math(root,state,profile,write=False)
 assert profile['actual_new_prefill_path_all_42_GateUp_verified'] and profile['verified_dense_batch']==8 and profile['LM_head_precision_in_verified_batch']=='BF16'
 assert profile['BM8_real_capture_verified'] and not profile['BM8_actual_C64_request_graph_replay_verified']
 result.update(final_profile_kind='BM8_M64_plus_qualified_M_range',controller_sources=state['controller_sources'],runtime_source_count=497,environment_count=62,fresh_Math_for_final_runtime_profile=True,previous_accuracy_not_reused=True,actual_C8_GateUp_NN_integer_path_verified=True,actual_C8_LM_BF16_path_verified=True,actual_prefill_range_fused_all_42_layers_verified=profile['actual_range_fused_all_42_layers_verified'],BM8_capture_verified_but_C8_not_actual_C64_replay=True,C64_supplementary_quality_required_before_official=True)
 if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def audit_benchmark(root,state,profile,quality,write=True):
 assert quality['runtime_source_count']==497 and quality['correct']>=100 and quality['fresh_Math_for_final_runtime_profile']
 frozen_check(quality['controller_sources']);frozen_check(state['controller_sources'])
 auxiliary=read(BASE/'C64_QUALITY_GATE_32g.json');assert auxiliary['complete'] and auxiliary['total']==105 and auxiliary['accuracy']>.95 and auxiliary['BM8_actual_C64_replay_verified']
 assert auxiliary['model_frozen_sources']==state['frozen_sources'] and auxiliary['only_admission_environment_difference'] and auxiliary['actual_prefill_range_fused_all_42_layers_verified']
 result=original_benchmark(root,state,profile,quality,write=False)
 result.update(final_profile_kind='BM8_M64_plus_qualified_M_range',runtime_source_count=497,environment_count=62,controller_sources=state['controller_sources'],fresh_official_Math8_correct=quality['correct'],fresh_C64_Math_correct=auxiliary['correct'],C64_supplementary_quality_source=str(BASE/'C64_QUALITY_GATE_32g.json'),C64_supplementary_quality_sha256=digest(BASE/'C64_QUALITY_GATE_32g.json'),previous_formal_total_tok_s=5695.95,vs_previous_formal_percent=100*(result['total_tok_s']/5695.95-1),historical_comparison_is_not_same_run_AB=True,BM8_actual_C64_replay_verified=True)
 if write:(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
