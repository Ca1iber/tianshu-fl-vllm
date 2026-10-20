# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_cold_full_dense_runtime_audit',BASE.parent/'tianshu_lm_precision_service_20261020/dense_audit.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==62 and state['profile_environment']['TIANSHU_DENSE_INT8_MODE']=='candidate'
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
    result.update(cold_start_mode="gpu4",full_62_environment_verified=True,packing_CPU_threads=during,CPU_threads_restored=True,all_43_banks_same_as_qualified_Math=True,loader_seconds=dense["load_model_seconds"],weight_packing_seconds=sum(r["packing_seconds"] for r in dense["banks"]),already_adopted_reflect_one_actually_executed=True)
    routes=list((root/"prefill_linear_records").glob("route_*.json"));assert len(routes)==1
    route=read(routes[0]);assert route["installed"] and route["mode"]=="candidate" and route["stage"]=="ready"
    assert route["activation_count"]==1 and route["activation_after_qualified_warmup"] and len(route["warmup_GateUp_M2048_prefixes"])==42
    assert route["policy_sha256"]==state["frozen_sources"][route["policy_source"]]
    for method in ("native_capture","native_linear"):
        item=route[method];assert item["sha256"]==state["frozen_sources"][item["source"]]
    active=[r for r in route["calls"].values() if r["stage"]=="ready" and not r["actually_inside_capture"] and r["M"]>=128]
    if state["prefill_int8_mode"]=="control":assert len({r["prefix"] for r in active})==42
    else:assert state["prefill_int8_mode"]=="candidate"
    expected="linear_contiguous"
    assert all(r["implementation"]==expected and r["calls"]>0 for r in active)
    assert all(r["M"] not in (8,64) for r in route["calls"].values())
    assert dense["lm_precision_mode"]==state["mode"]
    assert dense["supported_M_by_role"]["lm_head"]==([64] if state["mode"]=="candidate" else [8,64])
    if state["mode"]=="candidate":
        native=[r for r in dense["BF16_lm_launches"].values() if r["M"]==8 and not r["actually_inside_capture"]]
        assert sum(r["calls"] for r in native)>0
    result.update(prefill_linear_route=route,actual_ready_request_path_all_42_GateUp_verified=(len({r["prefix"] for r in active})==42),new_model_quality_not_yet_evaluated=True)
    return result
