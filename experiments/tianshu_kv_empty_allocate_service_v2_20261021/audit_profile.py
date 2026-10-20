# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_kv_empty_qualified_LM_audit',BASE.parent/'tianshu_lm_precision_service_20261020/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==62 and state['profile_environment']['TIANSHU_LM_PRECISION_MODE']=='candidate'
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    files=list((root/'kv_empty_allocate_records').glob('route_*.json'));assert len(files)==1
    row=read(files[0]);assert row['installed'] and row['mode']==state['mode']
    assert row['after_original_Scheduler_init'] and row['FutureKV_admission_wrapper_retained'] and row['only_inner_no_work_allocator_delegate_wrapped']
    assert row['actual_budget']==4096 and row['actual_max_seqs']==64 and row['model_precision_mode']=='candidate'
    assert row['actual_KV_blocks']==result['actual_KV_blocks'] and row['metrics_collector_is_None']
    assert row['no_KV_tensor_model_weight_RNG_or_request_data_modified'] and row['tracked_future_requests']==0 and row['future_gate_rejections']==0
    assert row['original_allocate_signature']==row['selected_allocate_signature']
    stats=row['stats'];assert stats['calls']>=100000 and stats['eligible']>=100000
    assert stats['calls']==stats['native_calls']+stats['shortcuts']==row['future_native_delegate_calls']
    assert stats['shortcuts']==(stats['eligible'] if state['mode']=='candidate' else 0)
    for field in ('policy','helper','original_native_allocate','future_policy'):
        path=row[field+'_source'];assert parent.parent.sha(path)==state['frozen_sources'][path]==row[field+'_sha256']
    result.update(mode=state['mode'],kv_empty_record=row,qualified_LM_model_precision_mode='candidate',
        original_FutureKV_native_calls_counter_counts_delegated_calls_including_shortcuts=True,new_model_Math_not_run=True)
    return result
