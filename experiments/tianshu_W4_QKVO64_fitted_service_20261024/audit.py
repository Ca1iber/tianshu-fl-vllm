# SPDX-License-Identifier: Apache-2.0
"""Audit 84 fitted banks and retain all original r4 runtime invariants."""
import importlib.util,json,sys
from pathlib import Path
from candidate_profile import digest,frozen_check,read
BASE=Path(__file__).resolve().parent
EXP=BASE.parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
lm=load('_QKVO64_preserved_LM_audit',EXP/'tianshu_W4_LMpad_service_v2_20261023/audit_profile.py')
down=load('_QKVO64_preserved_Down_audit',EXP/'tianshu_dotsaddz_Down64_service_v2_20261023/audit_profile.py')
math_validator=load('_QKVO64_fresh_math_validator',EXP/'tianshu_W4_O64_affine_service_20261024/audit.py')
# Reuse the full-105 validator, never an earlier model's scores or predictions.
audit_math=math_validator.audit_math

def one(root,folder,pattern='*.json'):
    files=list((root/folder).glob(pattern));assert len(files)==1,(folder,len(files))
    return read(files[0])

def audit_model(root,state,runtime=False):
    sources=state['frozen_sources'];frozen_check(sources)
    for p,h in sources.items():assert digest(root/'source_snapshot'/Path(p).relative_to('/'))==h
    preserved=lm.audit_startup(root,state);preserved.update(down.audit_startup(root,state))
    record=one(root,'QKVO64_fitted_records','route_*.json')
    assert record['complete'] and record['capture_completed'] and record['phase']=='runtime'
    assert record['load_complete'] and record['load_RNG_unchanged'] and record['mode']==state['mode']
    for key in ('public_register_kernel_used','original_backend_retained','original_BF16_W8_buffers_retained',
                'no_checkpoint_or_model_structure_change','no_new_apply_override','equal_payload_in_control_and_candidate','no_global_tensor_workspace'):
        assert record[key],key
    for pk,hk in [('source','source_sha256'),('scope_source','scope_sha256'),('quantizer_source','quantizer_sha256'),('original_init_source','original_init_sha256')]:
        assert digest(record[pk])==sources[record[pk]]==record[hk]
    selection_path=BASE/'SELECTED_BANKS_32g.json'
    assert record['selection_sha256']==digest(selection_path)
    selection={b['prefix']:b for b in read(selection_path)['banks']}
    assert {(b['role'],b['layer']) for b in selection.values()}=={(r,i) for r in ('O','QKV') for i in range(42)}
    assert len(record['banks'])==len(record['warmups'])==len(record['profile_probes'])==84
    assert {b['prefix'] for b in record['banks']}==set(selection)
    ids={b['bank_id'] for b in record['banks']};assert len(ids)==84
    total=0
    for b in record['banks']:
        expected=selection[b['prefix']];n=expected['N']
        assert n==(2560 if expected['role']=='QKV' else 2048) and expected['K']==2048
        assert b['layer']==expected['layer'] and b['original_sha256']==expected['original_sha256'] and b['original_weight_version_preserved']
        assert {e['name'] for e in b['entries']}=={'w','s','z'} and len(b['entries'])==3
        for e in b['entries']:
            shape,dtype,key={'w':([2048,n//8],'torch.int32','packed_sha256'),
                's':([16,n],'torch.bfloat16','scale_sha256'),'z':([16,n],'torch.bfloat16','zero_sha256')}[e['name']]
            assert e['shape']==shape and e['stride']==[shape[1],1] and e['dtype']==dtype
            assert e['sha256']==expected[key] and e['bytes']==shape[0]*shape[1]*(4 if e['name']=='w' else 2)
            total+=e['bytes']
        c=b['inside_DeviceMemoryProfiler']
        assert c['active'] and c['function']=='load_model' and sources[c['source']]==c['source_sha256']
    assert total==record['additional_packed_payload_bytes']==210567168
    assert record['own_buffers_in_actual_model_accounting']
    assert record['model_memory_usage_bytes']+16*1024**2>=record['all_model_unique_parameter_buffer_storage_bytes']
    for group,function in [('warmups','load_model'),('profile_probes','determine_available_memory')]:
        assert {p['bank_id'] for p in record[group]}==ids
        for p in record[group]:
            c=p['context'];assert p['M']==64 and p['zero_output'] and c['active'] and c['function']==function
            assert sources[c['source']]==c['source_sha256']
            if function=='determine_available_memory':assert c['manual_KV_bytes'] is None
    assert len(record['profile_windows'])==1
    for w in record['profile_windows']:
        assert w['RNG_unchanged'] and w['context']['active'] and w['context']['manual_KV_bytes'] is None
        assert w['peak_after']>=w['peak_before'] and w['peak_after']>=w['allocated_after']
    captured=[]
    for row in record['routes'].values():
        assert row['M']==64 and row['bank_id'] in ids and row['calls']>0
        assert row['role']==selection[row['prefix']]['role']
        if row['implementation']=='native_fitted_W4A16_QKVO64':
            assert state['mode']=='candidate' and row['eligible']
            assert row['context']==dict(mode='FULL',num_tokens=64,num_reqs=64,uniform=True,has_lora=False)
            if row['phase']=='capture' and row['capturing']:captured.append(row)
        else:assert row['implementation']=='unchanged_r4_backend'
    assert len({r['bank_id'] for r in captured})==(84 if state['mode']=='candidate' else 0)
    if state['mode']=='candidate':
        assert {(r['role'],selection[r['prefix']]['layer']) for r in captured}=={(r,i) for r in ('O','QKV') for i in range(42)}
    attention=one(root,'native_attention_records','route_*.json')
    assert attention['completed'] and attention['capture_completed'] and len(attention['banks'])==84
    assert attention['rows_by_role']=={'QKV':[128,512,4096],'O':[512,4096]}
    old=[r for r in attention['routes'].values() if r['M']==64 and r['phase']=='capture' and r['capturing']]
    assert {r['role'] for r in old}==(set() if state['mode']=='candidate' else {'QKV','O'})
    for role in ('QKV','O'):
        assert len({r['bank_id'] for r in old if r['role']==role})==(0 if state['mode']=='candidate' else 42)
    assert all(r['implementation']=='qualified_BF16_vendor_mm' for r in old)
    assert len({r['bank_id'] for r in attention['routes'].values() if r['phase']=='profile' and r['M']==4096 and r['implementation']=='native_A8W8_attention_projection'})==84
    fused=one(root,'nn_silu_fx_records')
    assert fused['completed'] and fused['all_42_M64_real_capture_routes_verified']
    assert len({r['prefix'] for r in fused['routes'].values() if r['M']==64 and r['actually_inside_capture'] and r['implementation']=='NN_two_dot_fused_SiLU' and r['component_binary_exact_match']})==42
    dense=one(root,'dense_int8_records','bindings_*.json')
    assert dense['load_complete'] and dense['actual_packed_banks']==43
    assert dense['packed_payload_bytes']==1326938112 and dense['original_BF16_weights_retained']
    assert not [r for r in dense['launches'].values() if r['role']=='lm_head' and 1<=r['M']<=64]
    assert not [r for r in dense['BF16_lm_launches'].values() if 1<=r['M']<=64]
    soft=one(root,'soft8_engine_records')
    assert soft['enabled'] and soft['layout']=='K8V16' and soft['model_dtype']=='torch.bfloat16'
    assert soft['max_model_len']==131072 and soft['prefix_cache'] and len(soft['value_plane_records'])==42
    arena=one(root,'small_arena_records')
    assert arena['mode']=='candidate' and arena['capacity_pages']==8192 and arena['scratch_bytes']==134234112
    assert arena['original_call_inherited'] and arena['original_over_capacity_fallback_unchanged']
    future=one(root,'future_kv_records')
    assert future['mode']==state['profile_environment']['TIANSHU_FUTURE_KV_MODE']
    assert future['running_allocations_forwarded'] and future['native_allocation_sizes_and_return_values_preserved']
    result=dict(complete=True,QKVO64_record=record,additional_payload_bytes=total,
        all84_original_and_packed_weight_hashes_verified=True,original944_model_sources_preserved=True,
        Down_LM_startup_audits_reexecuted=True,QKVO64_capture_banks=len({r['bank_id'] for r in captured}),
        QKV64_BF16_capture_banks=(0 if state['mode']=='candidate' else 42),O64_BF16_capture_banks=(0 if state['mode']=='candidate' else 42),
        original_GateUp_fusion_and_KV_profile_observed=True,model_memory_usage_bytes=record['model_memory_usage_bytes'],
        model_unique_storage_bytes=record['all_model_unique_parameter_buffer_storage_bytes'],physical_KV_blocks=future['KV_num_blocks'],runtime=runtime)
    if runtime:
        replay=one(root,'graph_segments_records','trace_*.json')
        graphs=[g for g in replay['graphs'].values() if 'num_tokens=64, num_reqs=64, uniform=True, has_lora=False' in g['descriptor'] and g['runtime_mode']=='FULL' and g['capture_calls']==1]
        actual=sum(g['replay_calls'] for g in graphs)
        if state['batch']==64:assert actual>=10
        current_lm=lm.audit_runtime(root,state,minimum_calls=10,batch=state['batch'])
        result.update(actual_FULL64_replays=actual,actual_LM_W4_batch_calls=current_lm['new_LM_verified_batch_W4_runtime_calls'],LM_W4_calls_by_M=current_lm['new_LM_W4_calls_by_M'])
        assert future['tracked_requests']==future['outstanding_blocks']==0
    return result
