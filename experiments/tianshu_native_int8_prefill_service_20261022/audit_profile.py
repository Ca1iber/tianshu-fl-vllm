# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def read(path):return json.loads(Path(path).read_text())
legacy=load('_native_inherited_450',BASE.parent/'tianshu_nn_silu_fx_screen_20261021/audit_profile.py')
adapted=load('_native_actual_prefill_audit',BASE/'prefill_audit.py');adapted.parent=legacy.prefill.parent;legacy.prefill=adapted

def audit_runtime(root,state,official=False):
    mode=state['mode'];assert mode in ('control','candidate')
    result=legacy.audit_runtime(root,dict(state,mode='candidate',native_mode=mode),official=official)
    native=read(next((root/'native_prefill_records').glob('route_*.json')));old=read(next((root/'m_range_prefill_records').glob('route_*.json')))
    assert native['mode']==mode and native['installed'] and native['completed'] and native['capture_completed'] and native['phase']=='runtime'
    assert native['no_new_weight_layout_or_payload'] and native['no_global_tensor_workspace'] and native['supported_rows']==[512,4096]
    assert native['load_warm_RNG_unchanged'] and len(native['load_warmups'])==2
    for row in native['load_warmups']:
        assert row['zero_output'] and row['inside_DeviceMemoryProfiler']['active'] and row['inside_DeviceMemoryProfiler']['sha256']==state['frozen_sources'][row['inside_DeviceMemoryProfiler']['source']]
        assert row['INT32_workspace_bytes']==row['M']*12288*4
    for name in ('source','kernel','quantizer','cache'):
        path=native[name] if name=='source' else native[name+'_source'];digest=native['source_sha256'] if name=='source' else native[name+'_sha256'];assert state['frozen_sources'][path]==digest
    cache=read(native['cache_source']);assert native['triton_cache_dir']==cache['qualified_model_cache_directory']
    observation=read(root/'observed_full_server_environment.json');full={**state['profile_environment'],**state['additional_environment']}
    assert len(state['profile_environment'])==62 and len(full)==63 and observation['environment']==full and observation['command']==state['server_command']
    assert state['additional_environment']=={'TRITON_CACHE_DIR':native['triton_cache_dir']}
    for m,row in native['native_kernels'].items():assert m in ('512','4096') and row['component_binary_exact_match'] and row['binary_sha256']==row['expected_binary_sha256']
    assert old['mode']=='candidate' and old['installed'] and old['completed'] and old['active'] and old['warm_RNG_unchanged'] and old['CPU_threads_restored']
    assert old['no_additional_packed_weight_payload'] and old['precision_M_whitelist_not_expanded'] and old['M64_existing_fusion_preserved'] and old['compile_sizes_unchanged']
    assert old['activated_after_qualified_capture_and_warm'] and old['fallback_activation_warm_before_capture_RNG_unchanged'] and old['fallback_activation_warm_performed']
    for key in ('policy','kernel','activation','quantizer','selection'):assert old[key+'_sha256']==state['frozen_sources'][old[key+'_source']]
    selection=read(BASE.parent/'tianshu_m_range_prefill_service_v2_20261021/selected_plan.json');component=read(selection['component_gate']);assert old['configuration_by_M']==selection['configuration_by_M'] and old['component_gate_sha256']==selection['component_gate_sha256']==state['frozen_sources'][selection['component_gate']]
    assert len(old['warmups'])==len(old['configuration_by_M']) and all(row['all_A_quantization_exact'] and row['CPU_selected_real_W_INT64_activation_passed'] and row['component_binary_exact_match'] for row in old['warmups'])
    assert set(old['actual_kernel_by_M'])==set(old['configuration_by_M'])
    for m,row in old['actual_kernel_by_M'].items():assert row['component_binary_exact_match'] and row['binary_sha256']==component['actual_timed_candidate_kernels'][m]['binary_sha256']
    for folder,replace in [('m_range_prefill_records',True),('native_prefill_records',True)]:
        rows=[read(path) for path in (root/folder/'fx_pass').glob('pass_*.json')];assert len(rows)==86
        dynamic=[r for r in rows if r['compile_range']==[1,4096]];static=[r for r in rows if r['compile_range']==[64,64]];assert len(dynamic)==len(static)==43
        assert sorted(i for r in dynamic for i in r['bank_ids'])==list(range(42)) and sum(r['replaced'] for r in dynamic)==42
        assert all(r['replaced']==0 and r['before_sha256']==r['after_sha256'] for r in static)
    active=[r for r in native['routes'].values() if r['phase']=='runtime' and r['implementation']=='native_A8W8_GateUp' and not r['capturing']]
    if mode=='candidate':
        assert len({r['bank_id'] for r in active if r['M']==4096})==42
        prof=[r for r in native['routes'].values() if r['phase']=='profile' and r['implementation']=='native_A8W8_GateUp' and r['M']==4096];assert len({r['bank_id'] for r in prof})==42
        assert any(w['context'] and w['context']['native_memory_context_active'] and not w['context']['manual_KV_bytes'] and w['native_calls_after']-w['native_calls_before']>=42 for w in native['profile_windows'])
        for row in active:assert row['M'] in (512,4096) and row['input_stride']==[2048,1] and row['INT32_workspace_bytes']==row['M']*12288*4 and row['binary_sha256']==native['native_kernels'][str(row['M'])]['binary_sha256']
    else:assert not active and all(r['implementation']=='delegate_existing_range' for r in native['routes'].values())
    original=read(BASE.parent/'tianshu_m_range_prefill_release_20261021/serve_profile_template.json')['frozen_sources'];assert len(original)==471 and all(state['frozen_sources'][p]==h for p,h in original.items())
    result.update(mode=mode,native_prefill_record=native,range_prefill_record=old,inherited_471_sources_exact_subset=True,legacy_62_environment_and_explicit_cache_environment_63_verified=True,all_inherited_attention_KV_scheduler_sampling_M8_M64_LM_checks_retained=True,no_fictitious_GEMM_routes=True,new_model_Math_required=True)
    return result
