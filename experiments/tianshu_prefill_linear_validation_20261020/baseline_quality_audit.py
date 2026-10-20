# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
p=BASE.parent/'tianshu_memory_combo_service_20261020/base_runtime_audit.py'
spec=importlib.util.spec_from_file_location('_future_quality_base_runtime',p)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
baseline_audit=base.audit_runtime

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit_runtime(root,state,official=False):
    result=baseline_audit(root,state,official=official)
    if state.get('workload',{}).get('input_tokens')==4096:assert result['zero_preemptions']
    rows=list((root/'long_segments_records').glob('bindings_*.json'))
    assert len(rows)==1
    record=read(rows[0]);expected='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106' if state['mode']=='candidate' else 'dc69f525d8c7686a8977f50e901af39820ce92486894890631c53fc01dc46700'
    assert record['installed'] and record['mode']==state['mode'] and record['actual_kernel_sha256']==expected
    assert sha(record['producer_source'])==sha(record['reducer_source'])==expected
    assert record['policy']==read(BASE/'batch_policy.json') and record['policy_sha256']==sha(BASE/'batch_policy.json')
    assert record['RNG_not_read_or_modified'] and record['unchanged_reader_python_dispatch'] and record['KV_format_writer_and_model_parameters_unchanged']
    files=list((root/'graph_segments_records').glob('trace_*.json'));assert len(files)==1
    trace=read(files[0]);assert set(trace['bound'])=={'graph','runner','routing'}
    for name,digest in trace['sources'].items():assert state['frozen_sources'][name]==digest and sha(name)==digest
    assert trace['attention_arguments_and_return_values_forwarded'] and trace['RNG_not_read_or_modified']
    captured=[x for x in trace['launches'].values() if x['actually_inside_capture'] and x['graph_key'] is not None]
    assert captured
    selected_captures=0
    for key,graph in trace['graphs'].items():
        if graph['replay_calls']:
            assert graph['capture_calls']>=1 and graph['graph_object_id'] is not None
            assert sum(x['calls'] for x in captured if x['graph_key']==key)>=42
    for row in captured:
        cfg=row['configuration']
        if cfg['IS_3D'] and 33<=cfg['sequences']<=56:
            assert cfg['segments']==(16 if state['mode']=='candidate' else 4)
            selected_captures+=row['calls']
    assert selected_captures>=42
    frames=[(int(n),v) for graph in trace['graphs'].values() for n,v in graph['frames'].items()]
    assert any(n>=(64 if official else 1) and v['max_computed']>=state.get('workload',{}).get('input_tokens',1) for n,v in frames)
    result.update(actual_FL_Graph_capture_and_replay_verified=True,selected_attention_captures=selected_captures,
                  graph_count=len(trace['graphs']),active_kernel_sha256=expected,mode=state['mode'])
    return result

long_graph_audit = audit_runtime

def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==61
    assert state['seq_limit']==64 and state['arena_mode']=='candidate'
    assert state['profile_environment']['TIANSHU_FUTURE_KV_MODE']=='candidate'
    assert 'TIANSHU_DECODE_METADATA_MODE' not in state['profile_environment']
    result=long_graph_audit(root,state,official=official)
    scheduler=[read(p) for p in (root/'seq_limit_records').glob('scheduler_*.json')]
    assert len(scheduler)==1
    row=scheduler[0]
    assert row['max_num_running_reqs']==row['config_max_num_seqs']==64
    assert row['max_num_batched_tokens']==row['max_num_scheduled_tokens']==4096
    assert row['async_scheduling'] is True and row['KV_watermark_blocks']==0
    arena=[read(p) for p in (root/'small_arena_records').glob('capacity_*.json')]
    assert arena and all(r['mode']=='candidate' and r['capacity_pages']==8192 for r in arena)
    rows=[read(p) for p in (root/'future_kv_records').glob('future_reservations_*.json')]
    assert len(rows)==1
    reservation=rows[0]
    assert reservation['installed'] and reservation['mode']=='candidate'
    assert reservation['native_allocator_sha256']=='2e91e77f84dfd9d4bed79f50392be72559207f54e845a3d3402baf88dc529320'
    assert reservation['native_allocator_sha256']==state['frozen_sources'][reservation['native_allocator_source']]==sha(reservation['native_allocator_source'])
    minimum=1024 if state['stage']=='official' else 105
    assert reservation['successful_admissions']>=minimum
    assert reservation['tracked_requests']==reservation['outstanding_blocks']==0
    result.update(future_reservation=reservation,native_scheduler_limit_verified=True,
        max_num_seqs=64,arena_capacity_pages=8192,actual_KV_blocks=row['KV_num_blocks'],
        fresh_quality_runtime_scope='C8 Math or original dual-workload official; no C64 assertion for Math',
        metadata_fusion_excluded=True,explicit_integer_metadata_experiment_not_adopted=True)
    return result

qualified_future_audit=audit_runtime

def audit_runtime(root,state,official=False):
    result=qualified_future_audit(root,state,official=official)
    required_batch=64 if official else 8
    required_context=4096 if official else 1
    rows=[read(p) for p in (root/'pair_load_records').glob('binding_*.json')]
    assert len(rows)==1;row=rows[0]
    assert row['installed'] and row['mode']==state['mode']
    assert row['parent_sha256']=='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
    assert row['candidate_sha256']=='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
    for name in ('parent','candidate','binding'):
        assert hashlib.sha256(Path(row[name+'_source']).read_bytes()).hexdigest()==row[name+'_sha256']
    assert state['frozen_sources'][row['binding_source']]==row['binding_sha256']
    spec=importlib.util.spec_from_file_location('_pair_audit_source',BASE.parent/'tianshu_k8v16_pair_load_20261020/source.py')
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    text,proof=source.render(row['parent_source'],'pair_rows')
    assert text==Path(row['candidate_source']).read_text()
    assert all(row['proof'][key]==value for key,value in proof.items())
    assert row['old_S16_binding_record_precedes_this_outer_wrapper'] and row['unchanged_writer_cache_V_values_and_reducer']
    assert row['RNG_not_read_or_modified'] and row['arguments_forwarded_unchanged']
    trace=read(next((root/'graph_segments_records').glob('trace_*.json')))
    launches=list(row['launches'].values());assert launches
    actual=0
    for launch in launches:
        selected=state['mode']=='candidate' and launch['IS_3D']
        assert launch['selected_pair_rows']==selected
        assert launch['active_source_sha256']==row['candidate_sha256' if selected else 'parent_sha256']
        assert launch['calls']>0
        if launch['IS_3D']:assert launch['warps']==launch['stages']==1
        if launch['inside_capture'] and launch['graph_key'] is not None:
            graph=trace['graphs'][launch['graph_key']]
            assert graph['capture_calls']>=1 and graph['graph_object_id'] is not None
            if any(int(n)==required_batch and frame['max_computed']>=required_context for n,frame in graph['frames'].items()):
                assert graph['replay_calls']>=(100 if official else 1)
                if launch['IS_3D'] and launch['sequences']==required_batch:
                    actual+=launch['calls']
    assert actual>=42
    result.update(pair_load_binding_verified=True,pair_load_record=row,
        inner_S16_record_scope='recorded before outer paired-load routing',
        active_decode_source_sha256=row['candidate_sha256' if state['mode']=='candidate' else 'parent_sha256'],
        actual_expected_batch_capture_and_model_replay_verified=True,verified_batch=required_batch)
    return result
