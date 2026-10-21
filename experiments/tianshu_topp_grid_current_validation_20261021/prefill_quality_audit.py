# SPDX-License-Identifier: Apache-2.0
"""C8 Math and C64 official evidence; exact same 438 runtime sources as screen."""
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
parent=load('_prefill_quality_existing_dense_audit',BASE/'dense_quality_audit.py')
read,sha=parent.read,parent.sha

def audit_runtime(root,state,official=False):
    result=parent.audit_runtime(root,state,official=official)
    assert len(state['frozen_sources'])==438 and len(state['profile_environment'])==62
    reference={r['prefix']:r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')}
    nnfiles=list((root/'int8_nn_gateup_records').glob('route_*.json'));assert len(nnfiles)==1;nn=read(nnfiles[0])
    assert nn['mode']=='candidate' and nn['installed'] and nn['completed'] and nn['after_original_capture'] and len(nn['banks'])==42
    assert nn['no_additional_packed_weight_payload'] and nn['logical_packed_banks_unchanged'] and nn['LM_route_unchanged']
    for b in nn['banks']:
        assert b['logical_shape']==[12288,2048] and b['logical_stride']==[1,12288] and b['payload_bytes']==12288*2048
        assert b['logical_sha256']==reference[b['prefix']]['packed_sha256'] and b['full_device_transpose_equal'] and b['full_logical_hash_equal'] and b['original_BF16_weight_version_preserved']
        assert b['inside_profiler']['active'] and b['inside_profiler']['sha256']==state['frozen_sources'][b['inside_profiler']['source']]
    for kind in ('policy','NT','NN'):assert nn[kind+'_sha256']==state['frozen_sources'][nn[kind+'_source']]==sha(nn[kind+'_source'])
    for m in (8,64):
        routes=[r for r in nn['routes'].values() if r['M']==m and r['actually_inside_capture']];assert len({r['prefix'] for r in routes})==42
        assert all(r['implementation']=='NN' and r['configuration']==nn['selected_configurations'][str(m)] and r['kernel_sha256']==nn['NN_sha256'] and r['calls']>0 for r in routes)
    files=list((root/'prefill_int8_records').glob('route_*.json'));assert len(files)==1;row=read(files[0])
    assert row['mode']=='candidate' and row['installed'] and row['completed'] and row['stage']=='active'
    assert row['same_NN_decode_in_both_arms'] and row['LM_not_intercepted'] and row['no_new_packed_weight_payload'] and len(row['banks'])==42
    assert row['after_original_capture'] and row['activated_after_qualified_capture_and_own_warmup'] and row['warm_CPU_GPU_RNG_unchanged'] and row['warm_all_declared_shapes_validated']
    assert row['warm_CPU_threads_before']==row['warm_CPU_threads_after'] and row['warm_CPU_threads_during']==4
    for kind in ('policy','kernel'):assert row[kind+'_sha256']==state['frozen_sources'][row[kind+'_source']]==sha(row[kind+'_source'])
    assert row['selection_sha256']==state['frozen_sources'][str(BASE.parent/'tianshu_prefill_int8_service_20261021/selected_plan.json')]
    assert set(map(int,row['warm_configs']))=={r['M'] for r in row['warmups']} and all(r['M']%128==0 and r['all_A_INT8_and_scales_exact'] and r['CPU_selected_real_W_INT64_to_BF16_passed'] for r in row['warmups'])
    actual=[r for r in row['routes'].values() if r['stage']=='active' and not r['actually_inside_capture'] and r['implementation']=='prefill_A8W8'];assert len({r['prefix'] for r in actual})==42
    if official:assert len({r['prefix'] for r in actual if r['M']==4096})==42
    for r in actual:
        assert r['M']>=128 and r['M']%128==0 and r['configuration']==row['warm_configs'][str(r['M'])] and r['input_stride']==[2048,1] and r['calls']>0
        assert r['kernel_sha256']==row['kernel_sha256']==state['frozen_sources'][r['kernel_source']]
    worker_pid=int(nnfiles[0].stem.split('_')[-1]);sql=read(root/f'sql_reflection_records/route_{worker_pid}.json')
    assert sql['reflection_scope']=='one_requested_table' and sql['configuration_values_modified'] is False and sql['table_exists']
    assert sql['helper_sha256']==state['frozen_sources'][str(BASE.parent/'tianshu_sql_workflow_20261014/reflect_one.py')]
    result.update(NN_gateup_record=nn,prefill_int8_record=row,legacy_dense_kernel_fields_scope='Original NT module only; NN capture routes and prefill runtime records identify actual kernel/config/binary.',
        actual_new_prefill_path_all_42_GateUp_verified=True,actual_prefill_INT8_shapes=sorted({r['M'] for r in actual}),actual_C8_Math_only_not_C64_request_claim=(not official),runtime_source_count=438)
    return result
