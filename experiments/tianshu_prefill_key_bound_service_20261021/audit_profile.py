# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
from candidate_profile import read,digest
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_prefill_key_bound_qualified_audit',BASE.parent/'tianshu_prefill_int8_service_20261021/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def audit_runtime(root,state,official=False):
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    files=list((root/'prefill_key_bound_records').glob('binding_*.json'));assert len(files)==1;record=read(files[0])
    assert record['mode']==state['mode'] and record['installed'] and record['capture_completed'] and record['actual_arena_pages']==8192
    assert record['only_prefill_bound_changed'] and record['worker_init_complete_before_install'] and record['no_parameter_or_KV_writer_or_sampling_changes']
    assert not record['additional_per_step_file_writes'] and record['existing_engine_records_only_for_actual_stage_evidence']
    for name in ('metadata_contract','native_route','policy','hook','component_gate'):
        p=record[name+'_source'];assert digest(p)==state['frozen_sources'][p]==record[name+'_sha256']
    engine_files=list(Path(state['profile_environment']['TIANSHU_SOFT8_RECORD_DIR']).glob('engine_*.json'));assert len(engine_files)==1
    engine=read(engine_files[0]);rows=[r for r in engine['dense_records'].values() if r.get('prefill_bound_stage')=='active']
    assert rows and all(r['prefill_bound_mode']==state['mode'] and r['actual_stage_pages']==r['sequences']*r['blocks_per_sequence'] for r in rows)
    assert all(r['stage_bound']<=r['original_global_maxK'] and r['actual_stage_pages']<=r['original_global_pages'] for r in rows)
    tight=[r for r in rows if r['only_prefill_stage_kwargs_copied']]
    if state['mode']=='control':assert not tight
    else:
        for r in tight:assert r['stage_bound']<r['original_global_maxK'] and r['sequences']>0
    result.update(mode=state['mode'],prefill_key_bound_binding=record,actual_ready_prefill_stage_rows=rows,
        actual_tightened_ready_rows=tight,actual_tightened_rows_observed=bool(tight),
        no_tightening_in_control=(not tight if state['mode']=='control' else None),
        fixed_arena8192_both_arms=True,new_model_Math_required_if_adopted=True)
    return result
