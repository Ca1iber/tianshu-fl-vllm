# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_cold_full_dense_runtime_audit',BASE/'dense_audit.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==61 and state['profile_environment']['TIANSHU_DENSE_INT8_MODE']=='candidate'
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
    sql=read(root/f'cold_start_records/sql/reflection_{worker_pid}.json')
    assert sql['installed'] and sql['mode']=='gpu4' and sql['original_sha256']==state['frozen_sources'][sql['original_source']]
    assert not sql['preimported_module'] and sql['target_reflection']==True and sql['total_calls']>0
    assert state['frozen_sources'][sql['hook_source']]==sql['hook_sha256'] and sql['original_class_other_methods_unchanged']
    result.update(cold_start_mode='gpu4',full_61_environment_verified=True,packing_CPU_threads=during,CPU_threads_restored=True,
        all_43_banks_same_as_qualified_Math=True,loader_seconds=dense['load_model_seconds'],weight_packing_seconds=sum(row['packing_seconds'] for row in dense['banks']),
        actual_worker_SQL_record=sql,actual_worker_SQL_calls_verified=True,model_quality_not_newly_evaluated=True)
    routes=list((root/"prefill_linear_records").glob("route_*.json"));assert len(routes)==1
    route=read(routes[0]);assert route["installed"] and route["mode"]==state["mode"] and route["stage"]=="ready"
    assert route["activation_count"]==1 and route["activation_after_qualified_warmup"] and len(route["warmup_GateUp_M2048_prefixes"])==42
    assert route["policy_sha256"]==state["frozen_sources"][route["policy_source"]]
    for method in ("native_capture","native_linear"):
        item=route[method];assert item["sha256"]==state["frozen_sources"][item["source"]]
    active=[r for r in route["calls"].values() if r["stage"]=="ready" and not r["actually_inside_capture"] and r["M"]>=128]
    assert len({r["prefix"] for r in active})==42
    expected="linear_contiguous" if state["mode"]=="candidate" else "qualified_mm"
    assert all(r["implementation"]==expected and r["calls"]>0 for r in active)
    assert all(r["M"] not in (8,64) for r in route["calls"].values())
    result.update(prefill_linear_route=route,actual_ready_request_path_all_42_GateUp_verified=True,new_model_quality_not_yet_evaluated=True)
    return result
