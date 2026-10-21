# SPDX-License-Identifier: Apache-2.0
"""Same450 numerical sources, native admission only, full Math105 at batch64."""
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
VALIDATION=BASE.parent/'tianshu_nn_silu_fx_validation_20261021'
SERVICE=BASE.parent/'tianshu_nn_silu_fx_screen_20261021'
MATH8=Path('/workspace/benchmark-results/20261021_nn_silu_fx_math8_tianshu')
sys.path.insert(0,str(BASE.parent/'tianshu_original_command_20261016'))
from validation_profile import BENCH_SHA,audit_math as original_math,csv_rows,normalized_environment

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
parent=load('_C64_aux_canonical450',VALIDATION/'framework_profile.py')
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
base_quality=load('_C64_aux_attention_base',BASE.parent/'tianshu_prefill_int8_validation_20261021/baseline_quality_audit.py')
dense=load('_C64_aux_real_dense',BASE/'dense_quality_audit.py')
quality=load('_C64_aux_qualified_prefill',VALIDATION/'audit_profile.py')

def prepare_runtime(root,screen_root):
    command,environment,profile,sources=parent.prepare_runtime(root,screen_root)
    canonical=dict(profile);assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate'
    profile['TIANSHU_FUTURE_KV_MODE']='control';environment.update(profile)
    assert {k for k in profile if profile[k]!=canonical[k]}=={'TIANSHU_FUTURE_KV_MODE'}
    return command,environment,profile,sources

def prepare(root,screen_root):
    pre=read(BASE/'CPU_GATE_32g.json');assert pre['complete'];frozen_check(pre['sources'])
    canonical=read(MATH8/'result.json');accuracy=read(MATH8/'accuracy_audit.json')
    assert canonical['phase']=='complete' and accuracy['complete'] and accuracy['total']==105 and accuracy['accuracy']>.95
    command,environment,profile,sources=prepare_runtime(root,screen_root)
    assert sources==canonical['frozen_sources'] and command==canonical['server_command']
    paths=[*BASE.glob('*.py'),*VALIDATION.glob('*.py'),* (BASE.parent/'tianshu_prefill_int8_validation_20261021').glob('*.py'),BASE.parent/'tianshu_prefill_int8_validation_20261021/batch_policy.json',BASE/'CPU_GATE_32g.json',BASE/'PLAN_32g.md',BASE/'run_after_math8.sh']
    controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
    for p,h in controllers.items():
        dest=root/'controller_source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes());assert digest(dest)==h
    return command,environment,profile,sources

def audit_native_admission(root,state,official=False):
    assert not official and state['profile_environment']['TIANSHU_FUTURE_KV_MODE']=='control'
    result=base_quality.long_graph_audit(root,state,official=False)
    future=read(next((root/'future_kv_records').glob('future_reservations_*.json')))
    assert future['mode']=='control' and future['installed'] and future['only_full_attention_single_pool']
    assert future['native_allocator_sha256']=='2e91e77f84dfd9d4bed79f50392be72559207f54e845a3d3402baf88dc529320'==state['frozen_sources'][future['native_allocator_source']]
    assert future['native_allocation_sizes_and_return_values_preserved'] and future['running_allocations_forwarded'] and future['no_KV_data_tokens_or_RNG_access']
    assert future['gate_rejections']==0 and future['successful_admissions']>=105 and future['max_tracked_requests']>=33 and future['diagnostic_would_reject']>0
    assert future['tracked_requests']==future['outstanding_blocks']==0
    scheduler=read(next((root/'seq_limit_records').glob('scheduler_*.json')))
    assert scheduler['max_num_running_reqs']==scheduler['config_max_num_seqs']==64
    assert scheduler['max_num_batched_tokens']==scheduler['max_num_scheduled_tokens']==4096 and scheduler['async_scheduling']
    pair=read(next((root/'pair_load_records').glob('binding_*.json')))
    assert pair['installed'] and pair['mode']=='candidate' and pair['candidate_sha256']=='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
    assert pair['RNG_not_read_or_modified'] and pair['arguments_forwarded_unchanged'] and pair['unchanged_writer_cache_V_values_and_reducer']
    result.update(future_reservation=future,native_scheduler_limit_verified=True,native_admission_only=True,actual_KV_blocks=scheduler['KV_num_blocks'],max_num_seqs=64)
    return result

dense.parent=sys.modules[__name__] if __name__ in sys.modules else None
# Imported through spec without registration: explicit object avoids relying on sys.modules.
class NativeAdmissionAudit:
    audit_runtime=staticmethod(audit_native_admission)
dense.parent=NativeAdmissionAudit
quality.parent=dense

def audit_runtime(root,state,official=False):
    assert not official
    result=quality.audit_runtime(root,state,official=False)
    result.update(actual_C8_Math_only_not_C64_request_claim=False,actual_C64_fused_replay_verified=True,auxiliary_Math_batch=64,
                  official_Math8_not_replaced=True,numerical_model_source_count=450)
    return result

def audit_math(root,state,profile,write=True):
    result=original_math(root,state,profile,write=False)
    assert profile['actual_C64_fused_replay_verified'] and profile['verified_dense_batch']==64 and profile['LM_head_precision_in_verified_batch']=='INT8'
    canonical=read(MATH8/'result.json');assert state['frozen_sources']==canonical['frozen_sources'] and len(state['frozen_sources'])==450
    actual_env=normalized_environment(state['profile_environment'],root);expected_env=normalized_environment(canonical['profile_environment'],MATH8)
    assert {k for k in actual_env if actual_env[k]!=expected_env[k]}=={'TIANSHU_FUTURE_KV_MODE'}
    assert actual_env['TIANSHU_FUTURE_KV_MODE']=='control' and expected_env['TIANSHU_FUTURE_KV_MODE']=='candidate'
    assert state['server_command']==canonical['server_command']
    result.update(fresh_math_for_original_command_profile=False,supplementary_C64_quality=True,official_Math8_not_replaced=True,total=105,
                  actual_C64_fused_replay_verified=True,actual_C64_LM_INT8_exercised=True,model_frozen_sources=state['frozen_sources'],only_admission_environment_difference=True,
                  admission_difference='TIANSHU_FUTURE_KV_MODE candidate→control; native allocator preserved',controller_sources=state['controller_sources'],model_quality_bound_to_same450_source_map=True)
    if write:
        (root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
        archive={k:v for k,v in result.items() if k not in ('profile','frozen_sources','controller_sources')}
        archive.update(raw_root=str(root),runtime_profile_summary=dict(max_tracked_requests=profile['future_reservation']['max_tracked_requests'],KV_blocks=profile['actual_KV_blocks'],verified_dense_batch=64),verifier_source=str(BASE/'auxiliary_profile.py'),verifier_sha256=digest(__file__))
        (BASE/'C64_QUALITY_GATE_32g.json').write_text(json.dumps(archive,indent=2)+'\n')
    return result

def audit_benchmark(*args,**kwargs):raise AssertionError('This controller only runs supplementary quality, not benchmark')
