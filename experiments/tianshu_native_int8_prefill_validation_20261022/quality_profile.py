# SPDX-License-Identifier: Apache-2.0
"""Fresh C8/C64 Math and official runs with honest per-workload path coverage."""
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
SERVICE=EXP/'tianshu_native_int8_prefill_service_v2_20261022'
SCREEN=EXP/'tianshu_native_int8_prefill_results_v2_20261022'
sys.path.insert(0,str(EXP/'tianshu_original_command_20261016'))
from validation_profile import BENCH_SHA,audit_math as original_math,audit_benchmark as original_benchmark,csv_rows,normalized_environment,official_command

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
parent=load('_native_validation_profile',SERVICE/'candidate_profile.py')
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
q8=load('_native_quality_C8_all_inherited',BASE/'quality_runtime.py')
q64=load('_native_quality_C64_all_inherited',BASE/'quality_runtime.py')
aux=load('_native_quality_retained_C64_attention_dense',EXP/'tianshu_nn_silu_fx_c64_quality_20261021/auxiliary_profile.py');q64.parent=aux.dense
extra=load('_native_quality_old_range_structure',BASE/'range_runtime_audit.py')
structure=load('_native_quality_real_native_structure',BASE/'native_structure.py')
full=load('_native_quality_full_official_audit',SERVICE/'audit_profile.py')

def prepare(root,screen_root,batch=8):
    assert screen_root.resolve()==SCREEN.resolve() and batch in (8,64)
    gate=read(SCREEN/'SCREEN_GATE_32g.json');assert gate['complete'] and gate['candidate_passes_perf_screen']
    candidate=read(Path(gate['arms']['candidate']['raw_root'])/'result.json');assert candidate['phase']=='complete'
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert sources==candidate['frozen_sources'] and command==candidate['server_command']==official_command()
    assert normalized_environment(profile,root)==normalized_environment(candidate['profile_environment'],candidate['output_root'])
    if batch==64:
        assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate';profile['TIANSHU_FUTURE_KV_MODE']='control';environment.update(profile)
    controller_paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',SCREEN/'SCREEN_GATE_32g.json',SCREEN/'verify_pair_cpu.py',SCREEN/'PLAN_32g.md'}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and path.endswith('.py') and Path(path).is_relative_to(EXP):controller_paths.add(Path(path).resolve())
    controllers={str(p):digest(p) for p in controller_paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
    for p,h in controllers.items():
        target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
    return command,environment,profile,sources

def audit_runtime(root,state,official=False):
    if official:return full.audit_runtime(root,state,official=True)
    batch=state['quality_batch'];assert batch in (8,64)
    result=(q8 if batch==8 else q64).audit_runtime(root,state,official=False)
    old=extra.audit(root,state,require_fused=False);native=structure.audit_runtime(root,state,official=False)
    result.update(old);result.update(native)
    result['all_inherited_attention_KV_scheduler_sampling_M8_M64_LM_checks_retained']=True
    result['actual_C8_Math_only_not_C64_request_claim']=(batch==8)
    routes=native['native_prefill_record']['routes'].values();hits=[r for r in routes if r['phase']=='runtime' and not r['capturing'] and r['implementation']=='native_A8W8_GateUp']
    result.update(actual_quality_batch=batch,actual_native_prefill_bank_count=len({r['bank_id'] for r in hits}),actual_native_prefill_M_values=sorted({r['M'] for r in hits}),actual_C8_is_not_C64_claim=(batch==8),actual_C64_fused_replay_verified=(batch==64))
    return result

def audit_math(root,state,profile,write=True):
    result=original_math(root,state,profile,write=False);batch=state['quality_batch']
    assert profile['verified_dense_batch']==batch and profile['LM_head_precision_in_verified_batch']==('BF16' if batch==8 else 'INT8')
    assert profile['actual_new_prefill_path_all_42_GateUp_verified']
    if batch==64:assert profile['actual_C64_fused_replay_verified'] and profile['actual_native_prefill_bank_count']==42 and 4096 in profile['actual_native_prefill_M_values']
    result.update(quality_batch=batch,fresh_math_for_original_command_profile=(batch==8),supplementary_C64_quality=(batch==64),controller_sources=state['controller_sources'],additional_environment=state['additional_environment'],runtime_source_count=len(state['frozen_sources']),environment_count=63,previous_accuracy_not_reused=True,actual_native_prefill_bank_count=profile['actual_native_prefill_bank_count'],actual_native_prefill_M_values=profile['actual_native_prefill_M_values'],C8_not_claimed_as_C64_or_unhit_native_path=True)
    if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def audit_benchmark(root,state,profile,quality,write=True):
    math64root=Path(state['math64_root']);q64gate=read(math64root/'accuracy_audit.json');s64=read(math64root/'result.json')
    assert q64gate['complete'] and s64['phase']=='complete' and q64gate['quality_batch']==64 and q64gate['total']==105 and q64gate['accuracy']>.95 and q64gate['actual_native_prefill_bank_count']==42
    assert q64gate['frozen_sources']==state['frozen_sources'] and q64gate['server_command']==state['server_command']
    a=normalized_environment(q64gate['profile_environment'],math64root);b=normalized_environment(state['profile_environment'],root)
    assert {k for k in a if a[k]!=b[k]}=={'TIANSHU_FUTURE_KV_MODE'} and a['TIANSHU_FUTURE_KV_MODE']=='control' and b['TIANSHU_FUTURE_KV_MODE']=='candidate'
    assert quality['additional_environment']==q64gate['additional_environment']==state['additional_environment']
    frozen_check(q64gate['controller_sources']);frozen_check(quality['controller_sources']);frozen_check(state['controller_sources'])
    result=original_benchmark(root,state,profile,quality,write=False)
    result.update(runtime_source_count=len(state['frozen_sources']),environment_count=63,previous_formal_total_tok_s=5695.95,vs_previous_formal_percent=100*(result['total_tok_s']/5695.95-1),historical_comparison_is_not_same_run_AB=True,fresh_C8_correct=quality['correct'],fresh_C64_correct=q64gate['correct'],fresh_C64_all42_native_prefill_verified=True,Math8_native_path_bank_count=quality['actual_native_prefill_bank_count'],C64_quality_sha256=digest(math64root/'accuracy_audit.json'))
    if write:(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
