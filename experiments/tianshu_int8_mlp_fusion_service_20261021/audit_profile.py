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
    assert mode in ('control','candidate') and env['TIANSHU_MLP_FUSION_MODE']==mode
    assert env['TIANSHU_LM_PRECISION_MODE']==env['TIANSHU_PREFILL_LINEAR_MODE']=='candidate'
    result=qualified_audit_runtime(root,dict(state,mode='candidate'),official=official)
    dense=result['dense_int8_record'];assert dense['mlp_fusion_mode']==mode and len(dense['MLP_bindings'])==42
    assert dense['unsupported_M_preserves_original_GateUp_precision'] and dense['down_projection_unchanged'] and dense['fresh_ephemeral_buffers']
    for key,digest_key in [('fusion_source','fusion_source_sha256'),('fusion_kernel_source','fusion_kernel_sha256'),('padding_kernel_source','padding_kernel_sha256'),('original_LlamaMLP_forward_source','original_LlamaMLP_forward_sha256')]:
        path=dense[key];assert parent.sha(path)==state['frozen_sources'][path]==dense[digest_key]
    banks={row['bank_id']:row for row in dense['banks'] if row['role']=='gateup'}
    assert set(banks)=={r['bank_id'] for r in dense['MLP_bindings']}
    for binding in dense['MLP_bindings']:
        assert binding['prefix']==banks[binding['bank_id']]['prefix'] and binding['prefix']==binding['module']+'.gate_up_proj'
        assert binding['MLP_class'].endswith('.LlamaMLP')
        for key in ('selected_activation','installed_forward'):
            path=binding[key+'_source'];assert parent.sha(path)==state['frozen_sources'][path]==binding[key+'_sha256']
    choices={r['M']:r for r in read(BASE.parent/'tianshu_int8_gateup_silu_20261021/SELECTED_CONFIGURATIONS_32g.json')}
    trace=read(next((root/'graph_segments_records').glob('trace_*.json')))
    actual={8:set(),64:set()};binary_matches={}
    for row in dense['fused_launches'].values():
        m=row['M'];assert mode=='candidate' and m in (8,64) and row['bank_id'] in banks
        target=32 if m==8 else 64
        assert row['allocated_M']==target and row['allocated_q_shape']==[target,2048] and row['allocated_scale_shape']==[target]
        assert row['allocated_output_shape']==[target,6144] and row['returned_output_shape']==[m,6144]
        assert row['actual_ephemeral_buffer_bytes']==target*(2048+4+2*6144) and row['matrix_pointer_alignment_mod256']==[0]*6
        assert row['configuration']==choices[m]['configuration'] and row['fused_kernel_sha256']==dense['fusion_kernel_sha256'] and row['padding_kernel_sha256']==dense['padding_kernel_sha256']
        assert row['quantizer_cache_hash'] and row['compiled_cache_hash'] and row['compiled_binary_sha256'] and row['calls']>0
        binary_matches[str(m)]=row['compiled_binary_sha256']==choices[m]['actual_timed_compiled']['binary_sha256']
        if row['actually_inside_capture'] and row['graph_key'] is not None:
            graph=trace['graphs'][row['graph_key']];required=100 if m==64 else 1;context=4096 if m==64 else 128
            if graph['capture_calls']>=1 and graph['replay_calls']>=required and any(int(n)==m and frame['max_computed']>=context for n,frame in graph['frames'].items()):actual[m].add(row['prefix'])
    if mode=='candidate':
        assert all(len(v)==42 for v in actual.values())
        assert all(r['M'] not in (8,64) for r in dense['MLP_native_fallback_calls'].values())
    else:assert not dense['fused_launches']
    route=result['prefill_linear_route']
    selected=[dict(source=p,qualname=n) for p,n in sorted({(r['selected_activation_source'],r['selected_activation_qualname']) for r in dense['MLP_bindings']})]
    result.update(mode=mode,MLP_fusion_mode=mode,all_42_actual_LlamaMLP_modules_bound=True,
        actual_fused_capture_and_model_replay_prefix_counts={str(m):len(v) for m,v in actual.items()},actual_binary_matches_component=binary_matches,
        selected_activation_sources=selected,MLP_wrapper_changes_compile_boundary=True,new_Math_required=True,
        component_bitwise_scope='installed FlagGems scalar; no assertion of every previous compiled activation boundary',
        capture_memory={key:route[key] for key in ('torch_allocated_before_capture','torch_reserved_before_capture','torch_allocated_after_capture','torch_reserved_after_capture','process_peak_allocated_after_capture')})
    return result
