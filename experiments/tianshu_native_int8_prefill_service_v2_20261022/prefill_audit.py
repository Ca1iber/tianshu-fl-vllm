# SPDX-License-Identifier: Apache-2.0
"""Preserve all inherited precision/bank checks; inspect actual native or old fused routes."""
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'tianshu_prefill_int8_service_20261021'
spec=importlib.util.spec_from_file_location('_native_prefill_NN_audit',BASE.parent/'tianshu_int8_nn_gateup_service_20261021/audit_profile.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(path):return json.loads(Path(path).read_text())
def audit_runtime(root,state,official=False):
    result=parent.audit_runtime(root,dict(state,mode='candidate',prefill_int8_mode=state['mode']),official=official)
    files=list((root/'prefill_int8_records').glob('route_*.json'));assert len(files)==1;row=read(files[0])
    assert row['mode']=='candidate' and row['installed'] and row['completed'] and row['stage']=='active'
    assert row['same_NN_decode_in_both_arms'] and row['LM_not_intercepted'] and row['no_new_packed_weight_payload'] and len(row['banks'])==42
    assert row['after_original_capture'] and row['activated_after_qualified_capture_and_own_warmup'] and row['warm_CPU_GPU_RNG_unchanged'] and row['warm_all_declared_shapes_validated']
    assert row['warm_CPU_threads_before']==row['warm_CPU_threads_after'] and row['warm_CPU_threads_during']==4
    assert all(r['logical_shape']==[12288,2048] and r['logical_stride']==[1,12288] for r in row['banks']) and sum(r['payload_bytes'] for r in row['banks'])==1056964608
    for kind in ('policy','kernel'):assert row[kind+'_sha256']==state['frozen_sources'][row[kind+'_source']]
    selection=read(BASE/'selected_plan.json');assert row['selection_sha256']==state['frozen_sources'][str(BASE/'selected_plan.json')]
    assert 128 in map(int,row['warm_configs']) and 4096 in map(int,row['warm_configs'])
    assert len(row['warmups'])==len(row['warm_configs']) and all(r['all_A_INT8_and_scales_exact'] and r['CPU_selected_real_W_INT64_to_BF16_passed'] for r in row['warmups'])
    assert all(r['configuration']==row['warm_configs'][str(r['M'])] for r in row['warmups'])
    own=read(next((root/'m_range_prefill_records').glob('route_*.json')));native=read(next((root/'native_int8_prefill_records').glob('route_*.json')))
    fused=[r for r in own['routes'].values() if r['stage']=='active' and not r['capturing'] and r['implementation']=='M_range_two_dot_fused']
    actual_native=[r for r in native['routes'].values() if r['phase']=='runtime' and not r['capturing'] and r['implementation']=='native_A8W8_GateUp']
    assert own['mode']=='candidate' and native['mode']==state['native_mode']
    actual=fused+actual_native;assert len({r['prefix'] for r in actual})==42 and len({r['prefix'] for r in actual if r['M']==4096})==42
    if state['native_mode']=='candidate':assert len({r['prefix'] for r in actual_native if r['M']==4096})==42
    else:assert not actual_native and len({r['prefix'] for r in fused if r['M']==4096})==42
    for route in fused:
        assert str(route['M']) in own['configuration_by_M'] and route['configuration']==own['configuration_by_M'][str(route['M'])] and route['calls']>0
        assert route['kernel_sha256']==own['kernel_sha256']==state['frozen_sources'][route['kernel_source']] and route['binary_sha256']==own['actual_kernel_by_M'][str(route['M'])]['binary_sha256']
    active=[r for r in row['routes'].values() if r['stage']=='active' and not r['actually_inside_capture'] and r['M']>=128 and r['implementation']=='prefill_A8W8']
    for route in active:
        assert route['M'] in map(int,row['warm_configs']) and route['configuration']==row['warm_configs'][str(route['M'])] and route['input_stride']==[2048,1] and route['calls']>0
        assert route['kernel_sha256']==row['kernel_sha256']==state['frozen_sources'][route['kernel_source']]
    result.update(prefill_int8_record=row,prefill_precision_candidate=True,same_NN_decode_control=True,prefill_runtime_actual_fused_or_native_routes_verified=True,no_fictitious_original_GEMM_launches=True,new_model_Math_required=True)
    return result
