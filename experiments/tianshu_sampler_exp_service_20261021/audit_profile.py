# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_sampler_warm_qualified_LM_audit',BASE.parent/'tianshu_lm_precision_service_20261020/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_runtime(root,state,official=False):
    assert state['profile_environment']['TIANSHU_LM_PRECISION_MODE']=='candidate' and len(state['profile_environment'])==62
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    files=list((root/'sampler_exp_warm_records').glob('route_*.json'));assert len(files)==1
    row=read(files[0]);assert row['mode']==state['mode'] and row['completed'] and row['activation_count']==1 and row['capture_calls']==1
    assert row['actual_budget']==4096 and row['actual_max_seqs']==64 and row['model_precision_mode']=='candidate'
    assert row['after_original_capture_returned'] and row['default_RNG_unchanged'] and row['original_capture_return_preserved']
    assert row['CPU_RNG_before']==row['CPU_RNG_after'] and row['GPU_RNG_before']==row['GPU_RNG_after']
    for field in ('original_capture','policy','helper','native_operator'):
        path=row[field+'_source'];assert parent.parent.sha(path)==state['frozen_sources'][path]==row[field+'_sha256']
    warm=row['prewarm'];assert warm['complete']
    if state['mode']=='candidate':
        assert warm['CPU_RNG_unchanged'] and warm['default_GPU_RNG_unchanged'] and warm['dedicated_generator_only'] and len(warm['calls'])==192
        assert warm['offset_bases']==[0,2**31,2**32] and warm['batches']==list(range(1,65))
    else:assert warm['skipped_in_control'] and warm['calls']==[]
    result.update(mode=state['mode'],qualified_LM_model_precision_mode='candidate',sampler_warmup_record=row,
        actual_sampler_warmup_after_capture_verified=True,actual_default_RNG_preserved=True,new_model_Math_not_run=True)
    return result
