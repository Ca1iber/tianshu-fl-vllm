# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_debug_log_qualified_LM_audit',BASE.parent/'tianshu_lm_precision_service_20261020/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==62 and state['profile_environment']['TIANSHU_LM_PRECISION_MODE']=='candidate'
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    files=list((root/'debug_log_records').glob('route_*.json'));assert len(files)==1
    row=read(files[0]);assert row['completed'] and row['mode']==state['mode'] and row['capture_calls']==1 and row['sample_calls']>=2048
    assert row['actual_budget']==4096 and row['actual_max_seqs']==64 and row['model_precision_mode']=='candidate'
    assert row['after_original_capture_returned'] and row['original_capture_return_preserved'] and row['all_original_sample_arguments_and_results_forwarded']
    assert row['RNG_and_request_data_not_accessed'] and row['index_configuration_cache_unchanged_missing']
    change=row['threshold_change'];assert change['complete'] and change['mode']==state['mode'] and change['handlers_and_filters_preserved'] and change['INFO_WARNING_ERROR_CRITICAL_preserved']
    assert row['actual_logger']==change['after'] and change['before']['effective_level']==10
    level=10 if state['mode']=='control' else 20
    assert row['active_level']==level and change['after']['effective_level']==level and row['INFO_and_above_enabled']
    assert all(c['level']==0 and c['effective_level']==level for c in row['children']) and len(row['children'])==7
    for field in ('policy','helper','original_capture','original_sample','native_logger'):
        path=row[field+'_source'];assert parent.parent.sha(path)==state['frozen_sources'][path]==row[field+'_sha256']
    result.update(mode=state['mode'],debug_log_record=row,qualified_LM_model_precision_mode='candidate',
        exact_same_model_calculation_and_native_config_values=True,new_model_Math_not_run=True)
    return result
