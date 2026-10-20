# SPDX-License-Identifier: Apache-2.0
"""Fresh C8 quality or official C64 with the same selective dense W8A8 profile."""
import hashlib,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_dense_quality_baseline',BASE/'baseline_quality_audit.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit_runtime(root,state,official=False):
    assert state['mode']=='candidate' and len(state['profile_environment'])==62
    assert state['profile_environment']['TIANSHU_DENSE_INT8_MODE']=='candidate'
    result=parent.audit_runtime(root,state,official=official)
    files=list((root/'dense_int8_records').glob('bindings_*.json'));assert len(files)==1
    row=read(files[0]);assert row['installed'] and row['mode']=='candidate' and row['load_complete']
    for key in ('policy','kernel'):
        assert sha(row[key+'_source'])==state['frozen_sources'][row[key+'_source']]==row[key+'_sha256']
    assert len(row['banks'])==row['actual_packed_banks']==43 and row['packed_payload_bytes']==1326938112
    assert len(row['visited'])==43 and sum(x['role']=='gateup' for x in row['banks'])==42 and sum(x['role']=='lm_head' for x in row['banks'])==1
    assert len({x['prefix'] for x in row['banks']})==43 and sum(x['packed_bytes'] for x in row['banks'])==1326938112
    for item in row['visited']:
        evidence=item['inside_profiler'];assert evidence['profiler_not_exited'] and evidence['function']=='load_model'
        assert sha(evidence['source'])==state['frozen_sources'][evidence['source']]==evidence['sha256']
    for name,digest in row['native_method_sources'].items():assert sha(name)==state['frozen_sources'][name]==digest
    assert row['model_memory_usage_bytes']+16*1024**2>=row['parameter_device_bytes']+row['packed_payload_bytes']
    assert row['original_BF16_weights_retained'] and row['checkpoint_never_modified'] and row['embedding_not_packed'] and row['RNG_not_read_or_modified']
    batch,context,replays=(64,4096,100) if official else (8,1,1)
    trace=read(next((root/'graph_segments_records').glob('trace_*.json')))
    selected=set()
    plan=read(BASE.parent/'tianshu_dense_w8a8_20261020/MODEL_INTEGRATION_PLAN_32g.json')
    configurations={(r['case'],r['M']):r['configuration'] for r in plan['selected_configurations']}
    for launch in row['launches'].values():
        assert launch['calls']>0 and launch['kernel_sha256']==row['kernel_sha256'] and launch['configuration']==configurations[launch['role'],launch['M']]
        if launch['role']!='gateup' or launch['M']!=batch or not launch['actually_inside_capture'] or launch['graph_key'] is None:continue
        graph=trace['graphs'][launch['graph_key']]
        if graph['capture_calls']>=1 and graph['replay_calls']>=replays and any(int(n)==batch and frame['max_computed']>=context for n,frame in graph['frames'].items()):selected.add(launch['bank_id'])
    assert len(selected)==42,(batch,len(selected))
    lm=[x for x in row['launches'].values() if x['role']=='lm_head' and x['M']==batch and not x['actually_inside_capture']]
    if batch==8:
        native=[r for r in row["BF16_lm_launches"].values() if r["M"]==8 and not r["actually_inside_capture"]]
        assert not lm and sum(r["calls"] for r in native)>=replays
        assert all(r["input_dtype"]==r["weight_dtype"]==r["output_dtype"]=="torch.bfloat16" for r in native)
    else:assert sum(x["calls"] for x in lm)>=replays
    result.update(dense_int8_binding_verified=True,dense_int8_record=row,extra_packed_weight_bytes=1326938112,
        actual_model_loader_profiler_active_during_packing_verified=True,verified_dense_batch=batch,
        actual_42_GateUp_banks_capture_and_replay_verified=True,actual_LM_head_outside_model_Graph_calls_verified=True)
    before,during,after=[row[k] for k in ("cpu_threads_before_load","cpu_threads_during_load","cpu_threads_after_load")]
    assert row["startup_mode"]=="gpu4" and during==4 and before==after
    reference={r["prefix"]:r for r in read(BASE.parent/"tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json")}
    for bank in row["banks"]:
        assert bank["packing_method"]=="GPU" and bank["packing_CPU_threads"]==4
        assert all(bank[k]==reference[bank["prefix"]][k] for k in ("shape","original_BF16_sha256","packed_sha256","scales_sha256","packed_bytes"))
    route_files=list((root/"prefill_linear_records").glob("route_*.json"));assert len(route_files)==1
    route=read(route_files[0]);assert route["mode"]=="candidate" and route["stage"]=="ready" and route["activation_after_qualified_warmup"] and len(route["warmup_GateUp_M2048_prefixes"])==42
    assert route["policy_sha256"]==state["frozen_sources"][route["policy_source"]]
    for name in ("native_capture","native_linear"):
        item=route[name];assert item["sha256"]==state["frozen_sources"][item["source"]]
    actual=[r for r in route["calls"].values() if r["stage"]=="ready" and not r["actually_inside_capture"] and r["M"]>=128]
    assert len({r["prefix"] for r in actual})==42 and all(r["implementation"]=="linear_contiguous" and r["calls"]>0 for r in actual)
    assert all(r["M"] not in (8,64) for r in route["calls"].values())
    assert row["lm_precision_mode"]=="candidate" and row["supported_M_by_role"]=={"gateup":[8,64],"lm_head":[64]}
    result.update(LM_head_precision_in_verified_batch=("BF16" if batch==8 else "INT8"),actual_C8_LM_BF16_path_verified=(batch==8),LM_M64_integer_not_exercised_by_C8_Math=True)
    result.update(prefill_linear_route=route,actual_new_prefill_path_all_42_GateUp_verified=True,all_43_banks_same_as_qualified_Math=True,packing_CPU_threads_restored=True)
    return result
