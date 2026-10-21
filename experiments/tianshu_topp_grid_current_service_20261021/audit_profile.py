# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_grid_service_parent_audit',BASE.parent/'tianshu_prefill_int8_service_20261021/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit_runtime(root,state,official=False):
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    cpu=read(BASE/'CPU_GATE_32g.json');assert cpu['complete'] and state['frozen_sources']==cpu['profiles'][0]['sources']
    assert sha(root/'source_snapshot'/(BASE/'CPU_GATE_32g.json').relative_to('/'))==sha(BASE/'CPU_GATE_32g.json')
    files=list((root/'topp_grid_service_records').glob('policy_*.json'));assert len(files)==1
    row=read(files[0]);assert row['mode']==state['mode'] and row['installed'] and row['completed']
    assert row['warm_complete'] and row['native_context_returned'] and row['capture_calls']==1
    assert row['actual_budget']==4096 and row['actual_max_seqs']==64 and row['warm_both_arms_same']
    assert row['native_profile_entered_before_warm'] and row['warm_before_native_context_yield'] and row['no_manual_peak_reset']
    assert row['warm_default_RNG_unchanged'] and row['warm_CPU_RNG_before']==row['warm_CPU_RNG_after'] and row['warm_GPU_RNG_before']==row['warm_GPU_RNG_after']
    assert len(row['warm_calls'])==114 and {(r['batch'],r['candidate_geometry']) for r in row['warm_calls']}=={(b,c) for b in range(8,65) for c in (False,True)}
    assert 'determine_available_memory' in row['profile_warm_stack'] and row['warm_temporary_tensors_released']
    assert row['shared_sampler_scratch_shape']==[64,130560] and row['shared_sampler_scratch_bytes']==33423360
    assert row['native_sampler_buffer_bytes']==4*130560*row['native_sampler_buffer_shape'][0]
    assert row['native_activation_peak_covers_warm_peak'] and row['native_final_peak_activation_memory']+row['native_before_profile_peak']>=row['native_peak_after_warm_bytes']
    assert row['after_complete_parent_capture_returned'] and row['original_capture_return_preserved']
    assert row['runtime_candidate_geometry']==(state['mode']=='candidate')
    assert row['actual_GEO_sha256']==sha(row['actual_GEO_source'])==sha(root/'topp_grid_service_records/actual_generated_geo.py')=='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
    assert row['actual_GEO_sha256'] in state['frozen_sources'].values()
    for key in ('policy','context','kernel_route','original_memory_profile','original_capture'):
        assert row[key+'_sha256']==state['frozen_sources'][row[key+'_source']]
    assert row['selection_sha256']==state['frozen_sources'][str(BASE/'selected_plan.json')]
    calls=list((root/'topp_grid_service_records').glob('calls_*.json'));assert len(calls)==1
    observed=read(calls[0]);runtime=[r for r in observed['routes'].values() if r['stage']=='runtime']
    assert {8,64}.issubset({r['batch'] for r in runtime}) and all(r['python_calls']>0 for r in runtime)
    for r in runtime:
        supported=(8<=r['batch']<=64 and not r['topk'] and r['topp'])
        assert r['active']==(supported and state['mode']=='candidate')
        if r['active']:
            m=r['actual_candidate'];assert m['BLOCK_SIZE']==16384 and m['BLOCK_SIZE_TRUNC']==8192 and m['num_warps']==16 and m['num_stages']==1
            assert m['grid']==[min(r['batch'],2*min(r['batch'],16))]
            assert m['source_sha256']==row['actual_GEO_sha256'] and len(m['binary_sha256'])==64
    smoke=state['joint_C8_smoke'];assert smoke['untimed'] and smoke['requests']==8
    result.update(mode=state['mode'],both_arms_current_prefill_INT8=True,all_parent_audits_preserved=True,
        topp_grid_record=row,topp_grid_observed=observed,actual_sampler_runtime_batches=sorted({r['batch'] for r in runtime}),
        candidate_geometry_active=(state['mode']=='candidate'),native_sampler_warm_accounting_verified=True,new_model_Math_required=True)
    return result
