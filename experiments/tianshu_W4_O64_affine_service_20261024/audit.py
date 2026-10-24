# SPDX-License-Identifier: Apache-2.0
"""Independent O64 route/bytes/native-profile and full Math105 verification."""
import collections
import importlib.util
import json
from pathlib import Path
import sys
from candidate_profile import digest, frozen_check, read
BASE=Path(__file__).resolve().parent
EXP=BASE.parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
lm=load('_O64_preserved_LM_audit',EXP/'tianshu_W4_LMpad_service_v2_20261023/audit_profile.py')
down=load('_O64_preserved_Down_audit',EXP/'tianshu_dotsaddz_Down64_service_v2_20261023/audit_profile.py')

def one(root,folder,pattern='*.json'):
    files=list((root/folder).glob(pattern))
    assert len(files)==1,(folder,len(files))
    return read(files[0])

def audit_model(root,state,runtime=False):
    sources=state['frozen_sources']
    frozen_check(sources)
    for path,h in sources.items():
        assert digest(root/'source_snapshot'/Path(path).relative_to('/'))==h
    result=lm.audit_startup(root,state)
    result.update(down.audit_startup(root,state))
    record=one(root,'O64_affine_records','route_*.json')
    assert record['complete'] and record['capture_completed'] and record['phase']=='runtime'
    assert record['load_complete'] and record['load_RNG_unchanged']
    assert record['mode']==state['mode'] and record['public_register_kernel_used']
    assert record['original_backend_retained'] and record['original_BF16_W8_buffers_retained']
    assert record['no_checkpoint_or_model_structure_change'] and record['no_new_apply_override']
    assert record['equal_payload_in_control_and_candidate'] and record['no_global_tensor_workspace']
    for path_key,hash_key in [('source','source_sha256'),('scope_source','scope_sha256'),
                              ('quantizer_source','quantizer_sha256'),('original_init_source','original_init_sha256')]:
        assert digest(record[path_key])==sources[record[path_key]]==record[hash_key]
    assert record['selection_sha256']==digest(BASE/'SELECTED_O_32g.json')
    selection={b['prefix']:b for b in read(BASE/'SELECTED_O_32g.json')['banks']}
    assert len(record['banks'])==len(record['warmups'])==len(record['profile_probes'])==42
    assert {b['prefix'] for b in record['banks']}==set(selection)
    ids={b['bank_id'] for b in record['banks']}
    assert len(ids)==42
    total=0
    for b in record['banks']:
        expected=selection[b['prefix']]
        assert b['original_sha256']==expected['original_sha256'] and b['original_weight_version_preserved']
        assert len(b['entries'])==3
        for e in b['entries']:
            shape,dtype,key={'w':([2048,256],'torch.int32','packed_sha256'),
                            's':([16,2048],'torch.bfloat16','scale_sha256'),
                            'z':([16,2048],'torch.bfloat16','zero_sha256')}[e['name']]
            assert e['shape']==shape and e['stride']==[shape[1],1] and e['dtype']==dtype
            assert e['sha256']==expected[key] and e['bytes']==shape[0]*shape[1]*(4 if e['name']=='w' else 2)
            total+=e['bytes']
        c=b['inside_DeviceMemoryProfiler']
        assert c['active'] and c['function']=='load_model' and sources[c['source']]==c['source_sha256']
    assert total==record['additional_packed_payload_bytes']==93585408
    assert record['own_buffers_in_actual_model_accounting']
    assert record['model_memory_usage_bytes']+16*1024**2>=record['all_model_unique_parameter_buffer_storage_bytes']
    for group,function in [('warmups','load_model'),('profile_probes','determine_available_memory')]:
        assert {p['bank_id'] for p in record[group]}==ids
        for p in record[group]:
            c=p['context']
            assert p['M']==64 and p['zero_output'] and c['active'] and c['function']==function
            assert sources[c['source']]==c['source_sha256']
            if function=='determine_available_memory': assert c['manual_KV_bytes'] is None
    assert len(record['profile_windows'])==1
    for w in record['profile_windows']:
        assert w['RNG_unchanged'] and w['context']['active'] and w['context']['manual_KV_bytes'] is None
        assert w['peak_after']>=w['peak_before'] and w['peak_after']>=w['allocated_after']
    new_captures=[]
    for row in record['routes'].values():
        assert row['M']==64 and row['bank_id'] in ids and row['calls']>0
        if row['implementation']=='native_affine_W4A16_O64':
            assert state['mode']=='candidate' and row['eligible']
            c=row['context']
            assert c==dict(mode='FULL',num_tokens=64,num_reqs=64,uniform=True,has_lora=False)
            if row['phase']=='capture' and row['capturing']: new_captures.append(row)
        else: assert row['implementation']=='unchanged_r4_backend'
    assert len({r['bank_id'] for r in new_captures})==(42 if state['mode']=='candidate' else 0)
    attention=one(root,'native_attention_records','route_*.json')
    assert attention['completed'] and attention['capture_completed'] and len(attention['banks'])==84
    assert attention['rows_by_role']=={'QKV':[128,512,4096],'O':[512,4096]}
    captures=[r for r in attention['routes'].values() if r['M']==64 and r['phase']=='capture' and r['capturing']]
    assert {r['role'] for r in captures}==({'QKV'} if state['mode']=='candidate' else {'QKV','O'})
    assert len({r['bank_id'] for r in captures if r['role']=='QKV'})==42
    assert len({r['bank_id'] for r in captures if r['role']=='O'})==(0 if state['mode']=='candidate' else 42)
    assert all(r['implementation']=='qualified_BF16_vendor_mm' for r in captures)
    assert len({r['bank_id'] for r in attention['routes'].values() if r['phase']=='profile'
        and r['M']==4096 and r['implementation']=='native_A8W8_attention_projection'})==84
    fused=one(root,'nn_silu_fx_records')
    assert fused['completed'] and fused['all_42_M64_real_capture_routes_verified']
    assert len({r['prefix'] for r in fused['routes'].values() if r['M']==64 and r['actually_inside_capture']
        and r['implementation']=='NN_two_dot_fused_SiLU' and r['component_binary_exact_match']})==42
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
    result=dict(complete=True,O64_record=record,additional_payload_bytes=total,
        all42_original_and_packed_O_weight_hashes_verified=True,
        original944_model_sources_preserved=True,Down_LM_startup_audits_reexecuted=True,
        O64_capture_banks=len({r['bank_id'] for r in new_captures}),
        QKV64_BF16_capture_banks=42,original_GateUp_fusion_and_KV_profile_observed=True,
        model_memory_usage_bytes=record['model_memory_usage_bytes'],
        model_unique_storage_bytes=record['all_model_unique_parameter_buffer_storage_bytes'],
        physical_KV_blocks=future['KV_num_blocks'],runtime=runtime)
    if runtime:
        replay=one(root,'graph_segments_records','trace_*.json')
        graphs=[g for g in replay['graphs'].values() if 'num_tokens=64, num_reqs=64, uniform=True, has_lora=False' in g['descriptor']
            and g['runtime_mode']=='FULL' and g['capture_calls']==1]
        actual_replays=sum(g['replay_calls'] for g in graphs)
        if state['batch']==64: assert actual_replays>=10
        current_lm=lm.audit_runtime(root,state,minimum_calls=10,batch=state['batch'])
        result.update(actual_FULL64_replays=actual_replays,
            actual_LM_W4_batch_calls=current_lm['new_LM_verified_batch_W4_runtime_calls'],
            LM_W4_calls_by_M=current_lm['new_LM_W4_calls_by_M'])
        assert future['tracked_requests']==future['outstanding_blocks']==0
    return result

def audit_math(root,state):
    reports=list(root.glob('evalscope/*/reports/minicpm/math_500.json'))
    assert len(reports)==1
    rp=reports[0]; report=read(rp); run=rp.parents[2]
    vp=run/'reviews/minicpm/math_500_Level 3.jsonl'
    pp=run/'predictions/minicpm/math_500_Level 3.jsonl'
    reviews=[json.loads(line) for line in vp.read_text().splitlines()]
    predictions=[json.loads(line) for line in pp.read_text().splitlines()]
    assert len(reviews)==len(predictions)==report['num']==105
    assert len({r['index'] for r in reviews})==len({p['index'] for p in predictions})==105
    assert {r['index'] for r in reviews}=={p['index'] for p in predictions}
    execution=report['execution_summary']
    assert execution['requested']==execution['succeeded']==105 and execution['errored']==0 and not execution['incomplete']
    scores=[r['sample_score']['score'] for r in reviews]
    assert all(s['status']=='success' and s['value']['accuracy'] in (0,1) for s in scores)
    assert all(p['model_output']['error'] is None for p in predictions)
    correct=int(sum(s['value']['accuracy'] for s in scores))
    metric=next(m for m in report['metrics'] if m['identity']['name']=='accuracy')
    assert metric['num']==105 and abs(metric['score']-correct/105)<.000051
    return dict(complete=True,correct=correct,total=105,accuracy=correct/105,above95=correct/105>.95,
        batch=state['batch'],fresh_entire_105=True,all_caps_in_denominator=True,
        incorrect_indices=[r['index'] for r in reviews if not r['sample_score']['score']['value']['accuracy']],
        stop_reasons=dict(collections.Counter(p['model_output']['choices'][0]['stop_reason'] for p in predictions)),
        raw_files={str(p.relative_to(root)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in (rp,vp,pp)})
