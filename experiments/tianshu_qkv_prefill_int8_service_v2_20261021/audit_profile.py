# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_qkv_prefill_current_complete_audit',BASE.parent/'tianshu_prefill_int8_service_20261021/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(path):return json.loads(Path(path).read_text())
def audit_runtime(root,state,official=False):
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    paths=list((root/'qkv_prefill_int8_records').glob('route_*.json'));assert len(paths)==1;row=read(paths[0])
    assert row['mode']==state['mode'] and row['installed'] and row['completed'] and row['stage']=='active'
    assert all(row[k] for k in ('QKV_prefill_only','decode_unchanged','O_not_intercepted','original_BF16_weights_retained','packing_inside_actual_active_DeviceMemoryProfiler','unknown_M_or_layout_falls_back_without_new_INT8_JIT','after_original_capture','parent_prefill_int8_remains_candidate','warm_CPU_GPU_RNG_unchanged','warm_all_declared_shapes_validated','activated_after_qualified_capture_and_own_warmup'))
    for kind in ('policy','kernel'):assert row[kind+'_sha256']==state['frozen_sources'][row[kind+'_source']]
    assert row['selection_sha256']==state['frozen_sources'][str(BASE/'selected_plan.json')]
    assert row['checkpoint_signatures_sha256']==state['frozen_sources'][str(BASE/'CHECKPOINT_SIGNATURES_32g.json')]
    reference={r['prefix']:r for r in read(BASE/'CHECKPOINT_SIGNATURES_32g.json')['banks']}
    assert len(row['banks'])==42 and {r['prefix'] for r in row['banks']}==set(reference)
    candidate=state['mode']=='candidate';extra=220631040 if candidate else 0
    assert sum(r['payload_bytes'] for r in row['banks'])==row['actual_extra_packed_payload_bytes']==extra
    for bank in row['banks']:
        assert bank['signature']==reference[bank['prefix']] and bank['original_weight_unchanged']
        proof=bank['inside_profiler'];assert proof['active'] and proof['sha256']==state['frozen_sources'][proof['source']]
        assert bank['control_placeholder_alias_original']==(not candidate)
        if candidate:
            assert bank['payload_bytes']==5253120 and bank['actual_packed_shape']==[2048,2560] and bank['actual_packed_stride']==[2560,1] and bank['actual_packed_dtype']=='torch.int8'
        else:assert bank['payload_bytes']==0 and bank['actual_packed_dtype']=='torch.bfloat16'
    dense=result['dense_int8_record']
    assert row['actual_model_memory_usage_bytes']==dense['model_memory_usage_bytes']
    assert row['actual_model_memory_usage_bytes']+16*1024**2>=dense['parameter_device_bytes']+dense['packed_payload_bytes']+extra
    assert row['warm_CPU_threads_before']==row['warm_CPU_threads_after']
    assert row['native_prefill_capture_sizes']==sorted(map(int,result['prefill_int8_record']['warm_configs']))
    if candidate:
        assert row['warm_CPU_threads_during']==4 and len(row['warmups'])==len(row['warm_configs'])==len(row['native_prefill_capture_sizes'])
        for warm in row['warmups']:
            assert warm['all_A_INT8_and_scales_exact'] and warm['CPU_selected_real_W_INT64_to_BF16_passed'] and warm['configuration']==row['warm_configs'][str(warm['M'])]
    else:assert row['warmups']==[] and row['warm_configs']=={}
    for m in (8,64):
        routes=[r for r in row['routes'].values() if r['M']==m and r['actually_inside_capture']]
        assert len({r['prefix'] for r in routes})==42 and all(r['implementation']=='delegate_qualified' for r in routes)
    active=[r for r in row['routes'].values() if r['stage']=='active' and r['M']>=128 and not r['actually_inside_capture']]
    assert len({r['prefix'] for r in active})==42
    if candidate:
        actual=[r for r in active if r['implementation']=='prefill_A8W8'];assert len({r['prefix'] for r in actual if r['M']==4096})==42
        for r in actual:
            assert r['M'] in map(int,row['warm_configs']) and r['configuration']==row['warm_configs'][str(r['M'])] and r['input_stride']==[2048,1] and r['calls']>0
            assert r['kernel_sha256']==row['kernel_sha256']==state['frozen_sources'][r['kernel_source']]
    else:assert all(r['implementation']=='delegate_qualified' for r in active)
    result.update(mode=state['mode'],qkv_prefill_int8_record=row,qkv_extra_packed_weight_bytes=extra,
        extra_packed_weight_bytes=dense['packed_payload_bytes']+extra,only_QKV_prefill_candidate=candidate,
        QKV_decode_all_42_delegated_actual_capture=True,current_GateUp_prefill_remains_candidate=True,new_model_Math_required=True)
    return result
