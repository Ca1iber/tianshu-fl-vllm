# SPDX-License-Identifier: Apache-2.0
"""Fresh official Math8 then N256, exact screened450 runtime source binding."""
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
SERVICE=BASE.parent/'tianshu_nn_silu_fx_screen_20261021'
sys.path.insert(0,str(BASE.parent/'tianshu_original_command_20261016'))
from validation_profile import BENCH_SHA,audit_math as original_math,audit_benchmark as original_benchmark,csv_rows,normalized_environment,official_command

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
parent=load('_FX_validation_screened450',SERVICE/'candidate_profile.py')
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
quality_runtime=load('_FX_validation_quality_runtime',BASE/'audit_profile.py')
full_runtime=load('_FX_validation_full_runtime',SERVICE/'audit_profile.py')

def prepare_runtime(root,screen_root):
    assert screen_root.resolve()==SERVICE.resolve()
    gate=read(SERVICE/'SCREEN_GATE_32g.json');assert gate['complete'] and gate['candidate_passes_perf_screen'] and gate['candidate_vs_original407_percent']>.5 and gate['all_spreads_below1_percent']
    candidate=read(Path(gate['arms']['candidate']['raw_root'])/'result.json');assert candidate['phase']=='complete'
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(sources)==450 and sources==candidate['frozen_sources'] and len(profile)==62 and command==candidate['server_command']==official_command()
    expected={k:v.replace(candidate['output_root']+'/',str(root)+'/') for k,v in candidate['profile_environment'].items()};assert profile==expected
    return command,environment,profile,sources

def prepare(root,screen_root):
    pre=read(BASE/'CPU_GATE_32g.json');assert pre['complete'] and pre['runtime_source_count']==450;frozen_check(pre['sources'])
    command,environment,profile,sources=prepare_runtime(root,screen_root)
    paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'CPU_GATE_32g.json',BASE/'PLAN_32g.md',SERVICE/'SCREEN_GATE_32g.json',SERVICE/'ROUNDS_32g.json',BASE.parent/'tianshu_original_command_20261016/validation_profile.py']
    controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
    for p,h in controllers.items():
        dest=root/'controller_source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes());assert digest(dest)==h
    return command,environment,profile,sources

def audit_runtime(root,state,official=False):
    if official:return full_runtime.audit_runtime(root,state,official=True)
    return quality_runtime.audit_runtime(root,state,official=False)

def audit_math(root,state,profile,write=True):
    result=original_math(root,state,profile,write=False)
    assert profile['actual_new_prefill_path_all_42_GateUp_verified'] and profile['verified_dense_batch']==8 and profile['LM_head_precision_in_verified_batch']=='BF16'
    result.update(final_profile_kind='NN_M64_FX_two_dot_GateUp_SiLU_and_static_compile64',controller_sources=state['controller_sources'],runtime_source_count=450,environment_count=62,
                  fresh_Math_for_final_runtime_profile=True,previous_accuracy_not_reused=True,actual_C8_GateUp_NN_integer_path_verified=True,actual_C8_LM_BF16_path_verified=True,actual_prefill_INT8_all_42_layers_verified=True,
                  C64_quality_not_exercised_by_C8_Math=True,C64_supplementary_quality_required_before_official=True)
    if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def audit_benchmark(root,state,profile,quality,write=True):
    assert quality['controller_sources']==state['controller_sources'] and quality['runtime_source_count']==450
    auxiliary_path=BASE.parent/'tianshu_nn_silu_fx_c64_quality_20261021/C64_QUALITY_GATE_32g.json'
    auxiliary=read(auxiliary_path)
    assert auxiliary['complete'] and auxiliary['total']==105 and auxiliary['accuracy']>.95 and auxiliary['actual_C64_fused_replay_verified']
    assert auxiliary['model_frozen_sources']==state['frozen_sources'] and auxiliary['only_admission_environment_difference']
    result=original_benchmark(root,state,profile,quality,write=False)
    result.update(final_profile_kind='NN_M64_FX_two_dot_GateUp_SiLU_and_static_compile64',runtime_source_count=450,environment_count=62,controller_sources=state['controller_sources'],
                  fresh_official_Math8_correct=quality['correct'],fresh_C64_Math_correct=auxiliary['correct'],C64_supplementary_quality_source=str(auxiliary_path),C64_supplementary_quality_sha256=digest(auxiliary_path),
                  previous_formal_total_tok_s=5586.28,vs_previous_formal_percent=100*(result['total_tok_s']/5586.28-1),historical_comparison_is_not_same_run_AB=True)
    if write:(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
