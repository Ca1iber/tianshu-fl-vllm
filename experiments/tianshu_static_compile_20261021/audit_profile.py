# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
from candidate_profile import read,digest
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_static_compile_qualified_audit',BASE.parent/'tianshu_prefill_int8_service_20261021/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def audit_runtime(root,state,official=False):
    mode=state['mode'];expected=[] if mode=='control' else [8,64]
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    configs=[r for p in (root/'static_compile_records').glob('config_*.json') for r in read(p)]
    assert configs and all(r['mode']==mode and r['actual_compile_sizes']==expected and r['actual_budget']==4096 and r['actual_max_num_seqs']==64 for r in configs)
    assert any(r['eligible'] and r['before_compile_sizes'] is None and r['default_applied']==(mode=='candidate') for r in configs)
    for r in configs:
        assert r['original_EngineArgs_compile_sizes_restored'] and r['explicit_compile_sizes_preserved'] and r['configuration_set_before_native_deepcopy']
        assert r['original_arguments_and_return_value_forwarded'] and r['RNG_and_request_data_not_accessed']
        for source,field in (('native','native_source_sha256'),('policy','policy_sha256')):
            p=r[source+'_source'];assert digest(p)==state['frozen_sources'][p]==r[field]
    files=list((root/'static_compile_records').glob('backend_*.json'));assert len(files)==1;trace=read(files[0])
    assert trace['mode']==mode and trace['actual_compile_sizes']==expected and trace['capture_completed'] and trace['after_original_capture']
    for key in ('original_range_entry_identity_preserved','original_find_called_once','RNG_and_inputs_not_read_or_modified','Python_range_selection_calls_are_not_Graph_replays','timing_path_has_no_JSON_writes'):assert trace[key]
    for kind in ('native_backend','trace'):
        p=trace[kind+'_source'];assert digest(p)==state['frozen_sources'][p]==trace[kind+'_sha256']
    graph=read(next((root/'graph_segments_records').glob('trace_*.json')));coverage={}
    for m,context,replays in ((8,128,1),(64,4096,100)):
        selected=[r for r in trace['routes'].values() if r['runtime_shape']==m and r['actually_inside_capture'] and r['graph_key'] is not None and r['calls']>0]
        assert selected
        for row in selected:assert row['compiled'] and row['compile_sizes']==expected and row['selected_range']==([m,m] if mode=='candidate' else [1,4096])
        replayed=[r for r in selected if graph['graphs'][r['graph_key']]['replay_calls']>=replays and any(int(n)==m and frame['max_computed']>=context for n,frame in graph['graphs'][r['graph_key']]['frames'].items())]
        assert replayed
        assert {r['piece_index'] for r in replayed}=={r['piece_index'] for r in selected}
        coverage[str(m)]=dict(piece_indices=sorted({r['piece_index'] for r in selected}),actual_capture_and_replay_verified=True,range=([m,m] if mode=='candidate' else [1,4096]))
    result.update(mode=mode,static_compile_default_configs=configs,static_compile_trace=trace,static_compile_capture_replay_coverage=coverage,
        precision_and_operator_routing_unchanged=True,inductor_compile_sizes=expected,new_Math_required_if_adopted=True)
    return result
