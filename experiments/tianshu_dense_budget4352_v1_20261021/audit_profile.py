# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_cold_full_dense_runtime_audit',BASE/'dense_audit.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==64 and state['profile_environment']['TIANSHU_DENSE_INT8_MODE']=='candidate'
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
    result.update(cold_start_mode="gpu4",full_64_environment_verified=True,packing_CPU_threads=during,CPU_threads_restored=True,all_43_banks_same_as_qualified_Math=True,loader_seconds=dense["load_model_seconds"],weight_packing_seconds=sum(r["packing_seconds"] for r in dense["banks"]),already_adopted_reflect_one_actually_executed=True)
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
    assert dense["lm_precision_mode"]==state["mode"]
    assert dense["supported_M_by_role"]["lm_head"]==([64] if state["mode"]=="candidate" else [8,64])
    if state["mode"]=="candidate":
        native=[r for r in dense["BF16_lm_launches"].values() if r["M"]==8 and not r["actually_inside_capture"]]
        assert sum(r["calls"] for r in native)>0
    result.update(prefill_linear_route=route,actual_ready_request_path_all_42_GateUp_verified=True,new_model_quality_not_yet_evaluated=True)
    return result

qualified_audit_runtime=audit_runtime

def audit_runtime(root,state,official=False):
    mode=state['mode'];budget=4096 if mode=='control' else 4352
    assert state['profile_environment']['TIANSHU_DENSE_BUDGET_MODE']==mode and state['actual_budget']==budget
    result=qualified_audit_runtime(root,dict(state,mode='candidate'),official=official)
    rows=[row for p in (root/'dense_budget_records').glob('budget_*.json') for row in read(p)]
    assert rows and any(row['default_applied'] for row in rows)
    for row in rows:
        assert row['mode']==mode and row['inherited_wrappers_called_once'] and row['RNG_and_request_data_not_accessed']
        assert row['original_arguments_and_return_value_forwarded'] and row['explicit_budget_preserved']
        assert parent.sha(row['native_source'])==state['frozen_sources'][row['native_source']]==row['native_source_sha256']
        assert parent.sha(row['policy_source'])==state['frozen_sources'][row['policy_source']]==row['policy_sha256']
        if row['default_applied']:assert row['before_budget'] is None and row['inherited_budget']==4096 and row['after_budget']==budget and row['max_num_seqs']==64
    result.update(mode=mode,scheduler_budget=budget,budget_default_records=rows,
        inner_default_records_scope='4096 before outer default selection',
        actual_worker_and_scheduler_budget_verified=True,new_Math_required=True)
    return result

budget_audit_runtime=audit_runtime

def audit_runtime(root,state,official=False):
    result=budget_audit_runtime(root,state,official=official)
    routes=[read(p) for p in (root/'dense_budget_records').glob('hook_route_*.json')]
    assert routes and all(r['exact_one_hook_redirect'] and r['runpy_restored'] for r in routes)
    for route in routes:
        for path,digest in route['sources'].items():assert parent.sha(path)==state['frozen_sources'][path]==digest
    guards=[read(p) for p in (root/'dense_budget_records').glob('guard_worker_*.json')];assert len(guards)==1
    guard=guards[0];path=guard['policy_source']
    assert parent.sha(path)==state['frozen_sources'][path]==guard['policy_sha256']
    assert path==str(BASE/'long_policy.py') and guard['actual_budget']==result['scheduler_budget'] and guard['actual_max_num_seqs']==64 and guard['actual_long_mode']=='candidate'
    result.update(independent_long_guard_worker_binding=guard,exact_one_long_hook_redirect_verified=True)
    return result

sampler_budget_audit_runtime=audit_runtime

def audit_runtime(root,state,official=False):
    result=sampler_budget_audit_runtime(root,state,official=official)
    files=list((root/'sampler_exp_warm_records').glob('route_*.json'));assert len(files)==1
    row=read(files[0]);assert row['mode']=='candidate' and row['completed'] and row['activation_count']==1 and row['capture_calls']==1
    assert row['actual_budget']==result['scheduler_budget'] and row['actual_max_seqs']==64 and row['model_precision_mode']=='candidate'
    assert row['after_original_capture_returned'] and row['default_RNG_unchanged'] and row['original_capture_return_preserved']
    assert row['CPU_RNG_before']==row['CPU_RNG_after'] and row['GPU_RNG_before']==row['GPU_RNG_after']
    for field in ('original_capture','policy','helper','native_operator'):
        path=row[field+'_source'];assert parent.sha(path)==state['frozen_sources'][path]==row[field+'_sha256']
    warm=row['prewarm'];assert warm['complete'] and len(warm['calls'])==warm['dispatch_classes']==4
    assert warm['CPU_RNG_unchanged'] and warm['default_GPU_RNG_unchanged'] and warm['actual_entry_presence_checked']
    assert warm['all_preexisting_dispatch_entries_preserved'] and warm['N_value_not_in_native_dispatch_key']
    assert warm['representative_batch']==64 and warm['vocab']==130560
    assert {tuple(x['dns_classes']) for x in warm['calls']}=={('i32',a,b) for a in ('i32','i64') for b in ('i32','i64')}
    result.update(sampler_warmup_record=row,sampler_four_ABI_entries_primed_after_capture=True,default_RNG_preserved=True,
        both_budget_arms_same_sampler_warmup=True,sampler_config_numerics_need_fresh_Math=True)
    return result
