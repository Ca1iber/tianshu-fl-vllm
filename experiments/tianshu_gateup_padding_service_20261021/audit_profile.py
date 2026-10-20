# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_cold_full_dense_runtime_audit',BASE/'dense_audit.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==63 and state['profile_environment']['TIANSHU_DENSE_INT8_MODE']=='candidate'
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    dense=result['dense_int8_record'];assert dense['startup_mode']=='gpu4'
    before,during,after=[dense[key] for key in ('cpu_threads_before_load','cpu_threads_during_load','cpu_threads_after_load')]
    assert before==after and during==4
    assert all(row['packing_CPU_threads']==during and row['packing_method']=='GPU' for row in dense['banks'])
    reference={row['prefix']:row for row in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')}
    assert len(reference)==len(dense['banks'])==43
    for row in dense['banks']:
        expected=reference[row['prefix']]
        assert all(row[key]==expected[key] for key in ('shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes'))
    bindings=next((root/'dense_int8_records').glob('bindings_*.json'));worker_pid=int(bindings.stem.split('_')[-1])
    sql=read(root/f"sql_reflection_records/route_{worker_pid}.json")
    assert sql["reflection_scope"]=="one_requested_table" and sql["configuration_values_modified"] is False and sql["table_exists"]
    assert sql["helper_sha256"]==state["frozen_sources"][str(BASE.parent/"tianshu_sql_workflow_20261014/reflect_one.py")]
    result.update(cold_start_mode="gpu4",full_63_environment_verified=True,packing_CPU_threads=during,CPU_threads_restored=True,all_43_banks_same_as_qualified_Math=True,loader_seconds=dense["load_model_seconds"],weight_packing_seconds=sum(r["packing_seconds"] for r in dense["banks"]),already_adopted_reflect_one_actually_executed=True)
    routes=list((root/"prefill_linear_records").glob("route_*.json"));assert len(routes)==1
    route=read(routes[0]);assert route["installed"] and route["mode"]=="candidate" and route["stage"]=="ready"
    assert route["activation_count"]==1 and route["activation_after_qualified_warmup"] and len(route["warmup_GateUp_M2048_prefixes"])==42
    assert route["policy_sha256"]==state["frozen_sources"][route["policy_source"]]
    for method in ("native_capture","native_linear"):
        item=route[method];assert item["sha256"]==state["frozen_sources"][item["source"]]
    active=[r for r in route["calls"].values() if r["stage"]=="ready" and not r["actually_inside_capture"] and r["M"]>=128]
    assert len({r["prefix"] for r in active})==42
    expected="linear_contiguous"
    assert all(r["implementation"]==expected and r["calls"]>0 for r in active)
    assert all(r["M"] not in (8,64) for r in route["calls"].values())
    assert dense["lm_precision_mode"]=="candidate"
    assert dense["supported_M_by_role"]["lm_head"]==[64]
    if True:
        native=[r for r in dense["BF16_lm_launches"].values() if r["M"]==8 and not r["actually_inside_capture"]]
        assert sum(r["calls"] for r in native)>0
    result.update(prefill_linear_route=route,actual_ready_request_path_all_42_GateUp_verified=True,new_model_quality_not_yet_evaluated=True)
    return result

qualified_audit_runtime=audit_runtime

def audit_runtime(root,state,official=False):
    mode=state['mode'];env=state['profile_environment']
    assert mode in ('control','candidate') and env['TIANSHU_TILE_PADDING_MODE']==mode
    assert env['TIANSHU_LM_PRECISION_MODE']==env['TIANSHU_PREFILL_LINEAR_MODE']=='candidate'
    result=qualified_audit_runtime(root,dict(state,mode='candidate'),official=official)
    dense=result['dense_int8_record'];assert dense['padding_mode']==mode
    assert dense['LM_precision_policy_unchanged'] and dense['fresh_buffers_per_custom_op'] and dense['returned_valid_rows_only']
    pad_path=dense['padding_kernel_source']
    assert parent.sha(pad_path)==state['frozen_sources'][pad_path]==dense['padding_kernel_sha256']
    expected=list(range(8,65)) if mode=='candidate' else [8,64]
    assert dense['supported_M_by_role']=={'gateup':expected,'lm_head':[64]}
    trace=read(next((root/'graph_segments_records').glob('trace_*.json')))
    active_prefixes={};active_M=set()
    for row in dense['launches'].values():
        selected=mode=='candidate' and row['role']=='gateup';m=row['M'];allocated=(32 if m<=32 else 64) if selected else m
        assert row['padding_selected']==selected and row['allocated_M']==allocated and row['actually_has_padding']==(m!=allocated)
        assert row['allocated_q_shape']==[allocated,row['K']] and row['allocated_scale_shape']==[allocated]
        assert row['allocated_output_shape']==[allocated,row['N']] and row['returned_output_shape']==[m,row['N']]
        assert row['actual_ephemeral_buffer_bytes']==allocated*(row['K']+4+2*row['N'])
        assert row['matrix_pointer_alignment_mod256']==[0]*6
        expected_sha=dense['padding_kernel_sha256'] if selected and allocated!=m else dense['kernel_sha256']
        assert row['quantizer_source_sha256']==expected_sha and row['quantizer_cache_hash']
        if selected:
            assert row['configuration']['bm']==allocated and row['configuration']['bn']==(128 if allocated==32 else 256)
        if selected and allocated!=m and row['actually_inside_capture'] and row['graph_key'] is not None:
            graph=trace['graphs'][row['graph_key']]
            if graph['capture_calls']>=1 and graph['replay_calls']>0 and any(frame['max_computed']>=state['workload']['input_tokens'] for frame in graph['frames'].values()):
                active_prefixes.setdefault(m,set()).add(row['prefix']);active_M.add(m)
    assert all(len(v)==42 for v in active_prefixes.values())
    route=result['prefill_linear_route']
    for key in ('torch_allocated_before_capture','torch_reserved_before_capture','torch_allocated_after_capture','torch_reserved_after_capture','process_peak_allocated_after_capture'):assert route[key]>0
    result.update(mode=mode,tile_padding_mode=mode,all_padding_launch_allocations_and_alignment_verified=True,
        padding_actual_4K_replayed_M=sorted(active_M),padding_actual_4K_replayed_prefix_counts={str(k):len(v) for k,v in active_prefixes.items()},
        new_padding_actually_executed_in_timed_context=bool(active_M),GateUp_minimum_INT8_M=8,
        maximum_recorded_GateUp_ephemeral_buffer_bytes=max(r['actual_ephemeral_buffer_bytes'] for r in dense['launches'].values() if r['role']=='gateup'),
        capture_memory={k:route[k] for k in ('torch_allocated_before_capture','torch_reserved_before_capture','torch_allocated_after_capture','torch_reserved_after_capture','process_peak_allocated_after_capture')})
    if mode=='control':assert not active_M
    return result
