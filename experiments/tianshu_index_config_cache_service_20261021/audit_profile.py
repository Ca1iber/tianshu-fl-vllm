# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_index_cache_qualified_LM_audit',BASE.parent/'tianshu_lm_precision_service_20261020/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==62 and state['profile_environment']['TIANSHU_LM_PRECISION_MODE']=='candidate'
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    files=list((root/'index_config_cache_records').glob('route_*.json'));assert len(files)==1
    row=read(files[0]);assert row['completed'] and row['mode']==state['mode'] and row['capture_calls']==1
    assert row['actual_budget']==4096 and row['actual_max_seqs']==64 and row['model_precision_mode']=='candidate'
    assert row['after_original_capture_returned'] and row['original_capture_return_preserved']
    assert row['cache_entry_initially_missing'] and row['RNG_and_request_data_not_accessed'] and row['native_config_values_not_modified']
    assert row['lookup_calls']>=2048 and row['configuration_count']==17
    assert row['last_signature_matches_CPU_proof'] and row['preexisting_cache_entry_identities_preserved']
    component=BASE.parent/'tianshu_index_config_cache_20261021';gate=read(component/'CPU_GATE_32g.json')
    assert row['configuration_signature']==gate['full_config_signature']
    for field in ('policy','helper','original_capture','native_config_loader'):
        path=row[field+'_source'];assert parent.parent.sha(path)==state['frozen_sources'][path]==row[field+'_sha256']
    candidate=state['mode']=='candidate'
    assert row['actual_entry_present']==row['actual_result_is_cached_entry']==candidate
    assert row['construction_misses']==(0 if candidate else row['lookup_calls'])
    if candidate:assert row['prime']['complete'] and row['prime']['configuration_count']==17 and row['prime']['signature']==gate['full_config_signature']
    result.update(mode=state['mode'],index_config_cache_record=row,qualified_LM_model_precision_mode='candidate',
        exact_same_native_configuration_values_verified=True,new_model_Math_not_run=True)
    return result
