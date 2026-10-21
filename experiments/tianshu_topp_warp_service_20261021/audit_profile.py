# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_topp_warp_current_parent_audit',BASE.parent/'tianshu_prefill_int8_service_20261021/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit_runtime(root,state,official=False):
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    cpu=read(BASE/"CPU_GATE_32g.json");assert cpu["complete"] and len(state["frozen_sources"])==cpu["runtime_source_count"] and state["frozen_sources"]==cpu["profiles"][0]["sources"]
    assert digest(root/"source_snapshot"/(BASE/"CPU_GATE_32g.json").relative_to("/"))==digest(BASE/"CPU_GATE_32g.json")
    files=list((root/'topp_warp_records').glob('route_*.json'));assert len(files)==1;row=read(files[0])
    assert row['mode']==state['mode'] and row['installed'] and row['completed'] and row['active']
    assert row['profile_warm_complete'] and row['native_context_returned'] and row['capture_calls']==1
    assert row['actual_budget']==4096 and row['actual_max_seqs']==64
    assert row['warm_both_arms_same'] and row['warm_default_RNG_unchanged'] and row['warm_temporary_tensors_released']
    assert row['warm_CPU_RNG_before']==row['warm_CPU_RNG_after'] and row['warm_GPU_RNG_before']==row['warm_GPU_RNG_after']
    assert len(row['warm_calls'])==114 and {(x['batch'],x['num_warps']) for x in row['warm_calls']}=={(b,w) for b in range(8,65) for w in (4,16)}
    assert all(x['input_stride']==[130560,1] for x in row['warm_calls'])
    assert 'determine_available_memory' in row['profile_warm_stack'] and row['native_activation_peak_covers_warm_peak']
    assert row['native_final_peak_activation_memory']+row['native_before_profile_peak']>=row['native_peak_after_warm_bytes']
    assert row['after_complete_parent_capture_returned'] and row['original_capture_return_preserved'] and row['request_ready_seen']
    assert row['native_sampler_buffer_shape'][1]==130560 and 8<=row['native_sampler_buffer_shape'][0]<=64
    assert row['native_sampler_buffer_bytes']==4*130560*row['native_sampler_buffer_shape'][0]
    for field in ('policy','route','kernel','original_memory_profile','original_capture'):
        path=row[field+'_source'];assert row[field+'_sha256']==state['frozen_sources'][path]
    assert row['selection_sha256']==state['frozen_sources'][str(BASE/'selected_plan.json')]
    active=[x for x in row['routes'].values() if x['stage']=='runtime'];assert active and any(x['batch']==64 and x['calls']>0 for x in active)
    expected=16 if state['mode']=='candidate' else 4
    for x in active:
        assert x['num_warps']==expected and x['shape']==[x['batch'],130560] and x['stride']==[130560,1] and x['dtype']=='torch.float32'
        assert x['calls']>0 and not x['actually_inside_capture'] and x['TOPK_ENABLED'] is False and x['TOPP_ENABLED'] is True and x['BLOCK_SIZE']==8192 and x['BLOCK_SIZE_TRUNC']==4096
    result.update(mode=state['mode'],both_arms_current_prefill_INT8_recommended=True,topp_warp_record=row,actual_topp_runtime_warps=expected,topp_FP32_reduction_differences_expected=True,all_parent_audits_preserved=True,topp_warm_native_profile_verified=True,new_model_Math_required=True)
    return result
