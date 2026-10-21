# SPDX-License-Identifier: Apache-2.0
"""Retain full qualified audit; explicitly audit true fused M64 routes."""
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def read(path):return json.loads(Path(path).read_text())
prefill=load('_fx_screen_prefill_full_audit',BASE.parent/'tianshu_prefill_int8_service_20261021/audit_profile.py')
nn=load('_fx_screen_NN_audit',BASE/'nn_audit.py')
lower=load('_fx_screen_prefill_qualified',BASE.parent/'tianshu_prefill_int8_service_20261021/adapted_qualified_audit.py')
dense=load('_fx_screen_dense_audit',BASE/'dense_audit.py')
lower.parent=dense;nn.parent=lower;prefill.parent=nn

def audit_runtime(root,state,official=False):
    mode=state['mode'];assert mode in ('control','candidate')
    result=prefill.audit_runtime(root,dict(state,mode='candidate',fusion_mode=mode),official=official)
    fusion=read(next((root/'nn_silu_fx_records').glob('route_*.json')))
    assert fusion['mode']==mode and fusion['installed'] and fusion['completed'] and fusion['all_42_M64_real_capture_routes_verified'] and fusion['warm_before_capture_RNG_unchanged']
    assert fusion['no_additional_packed_weight_payload'] and fusion['M8_decode_unchanged'] and fusion['dynamic_prefill_graph_unchanged']
    for name in ('policy','kernel','quantizer'):assert fusion[name+'_sha256']==state['frozen_sources'][fusion[name+'_source']]
    rows=[read(p) for p in (root/'nn_silu_fx_records/fx_pass').glob('pass_*.json')];assert len(rows)==86
    static=[r for r in rows if r['compile_range']==[64,64]];dynamic=[r for r in rows if r['compile_range']==[1,4096]]
    assert len(static)==len(dynamic)==43
    assert sorted(i for r in static for i in r['bank_ids'])==list(range(42))
    assert sum(r['replaced'] for r in static)==(42 if mode=='candidate' else 0)
    assert all(r['replaced']==0 and r['before_sha256']==r['after_sha256'] and r['no_other_ranges_changed'] for r in dynamic)
    if mode=='control':assert all(r['before_sha256']==r['after_sha256'] for r in static) and not fusion['routes']
    else:
        routes=[r for r in fusion['routes'].values() if r['M']==64 and r['actually_inside_capture']]
        assert len({r['prefix'] for r in routes})==42
        trace=read(next((root/'graph_segments_records').glob('trace_*.json')));selected=set()
        for r in routes:
            assert r['configuration']==fusion['configuration'] and r['kernel_sha256']==state['frozen_sources'][r['kernel_source']] and r['graph_key'] is not None
            graph=trace['graphs'][r['graph_key']]
            if graph['capture_calls']>=1 and graph['replay_calls']>=100 and any(int(n)==64 and f['max_computed']>=4096 for n,f in graph['frames'].items()):selected.add(r['prefix'])
        assert len(selected)==42
    warm=read(next((root/'index_warm_records').glob('warm_*.json')))
    assert warm['mode']==mode and warm['installed'] and warm['completed'] and warm['CPU_GPU_RNG_unchanged'] and warm['request_time_index_not_wrapped']
    assert {(r['input_shape'][0],r['index_shape'][0]) for r in warm['rows']}=={(63,32),(63,4001),(64,32),(64,4001)}
    assert all(r['all_outputs_CPU_exact'] and r['before_original_capture'] for r in warm['rows'])
    for name in ('policy','index'):assert warm[name+'_sha256']==state['frozen_sources'][warm[name+'_source']]
    original=read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/RUNTIME_BINDING_GATE_32g.json')['frozen_sources']
    assert len(original)==407 and all(state['frozen_sources'][p]==h for p,h in original.items())
    log=(root/'server.log').read_text(errors='replace');assert log.count('Application startup complete.')==1
    assert 'Triton autotuning for function' not in log.split('Application startup complete.',1)[1]
    result.update(mode=mode,fusion_mode=mode,prefill_mode='candidate',same_NN_decode_control=(mode=='control'),physical_NN_layout_same_in_all_arms=True,
                  original_407_source_map_exact_subset=True,native_dynamic_graphs_preserved=True,M64_FX_fusion_record=fusion,
                  legacy_prefill_same_NN_field_scope='Original prefill policy precedes outer FX fusion; M64 actual kernel is verified here in M64_FX_fusion_record.',
                  full_inherited_attention_KV_scheduler_weights_LM_prefill_checks_retained=True,new_model_Math_required=True)
    return result
