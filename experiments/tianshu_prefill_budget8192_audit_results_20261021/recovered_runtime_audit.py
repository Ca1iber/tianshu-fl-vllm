# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path('/workspace/vllm-plugin-FL/experiments/tianshu_prefill_budget8192_service_20261021')
spec=importlib.util.spec_from_file_location('_prefill_NN_decoder_audit',BASE.parent/'tianshu_prefill_budget8192_nn_20261021/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
spec=importlib.util.spec_from_file_location('_prefill_explicit_qualified_audit',BASE/'adapted_qualified_audit.py')
adapted=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapted);parent.parent=adapted

def read(p):return json.loads(Path(p).read_text())
def audit_precision_runtime(root,state,official=False):
    result=parent.audit_runtime(root,dict(state,mode='candidate',prefill_int8_mode=state['mode']),official=official)
    files=list((root/'prefill_int8_records').glob('route_*.json'));assert len(files)==1;row=read(files[0])
    assert row['mode']==state['mode'] and row['installed'] and row['completed'] and row['stage']=='active'
    assert row['same_NN_decode_in_both_arms'] and row['LM_not_intercepted'] and row['no_new_packed_weight_payload'] and len(row['banks'])==42
    assert row['after_original_capture'] and row['activated_after_qualified_capture_and_own_warmup'] and row['warm_CPU_GPU_RNG_unchanged'] and row['warm_all_declared_shapes_validated']
    assert row['warm_CPU_threads_before']==row['warm_CPU_threads_after'] and row['warm_CPU_threads_during']==4
    assert all(r['logical_shape']==[12288,2048] and r['logical_stride']==[1,12288] for r in row['banks'])
    assert sum(r['payload_bytes'] for r in row['banks'])==1056964608
    for kind in ('policy','kernel'):assert row[kind+'_sha256']==state['frozen_sources'][row[kind+'_source']]
    selection=read(BASE/'selected_plan.json');assert row['selection_sha256']==state['frozen_sources'][str(BASE/'selected_plan.json')]
    assert 128 in map(int,row['warm_configs']) and 4096 in map(int,row['warm_configs'])
    assert len(row['warmups'])==len(row['warm_configs']) and all(r['all_A_INT8_and_scales_exact'] and r['CPU_selected_real_W_INT64_to_BF16_passed'] for r in row['warmups'])
    for r in row['warmups']:assert r['configuration']==row['warm_configs'][str(r['M'])]
    active=[r for r in row['routes'].values() if r['stage']=='active' and not r['actually_inside_capture'] and r['M']>=128]
    if state['mode']=='candidate':
        actual=[r for r in active if r['implementation']=='prefill_A8W8'];assert len({r['prefix'] for r in actual})==42
        assert len({r['prefix'] for r in actual if r['M']==state['actual_budget']})==42
        for r in actual:
            assert r['M'] in map(int,row['warm_configs']) and r['configuration']==row['warm_configs'][str(r['M'])] and r['input_stride']==[2048,1] and r['calls']>0
            assert r['kernel_sha256']==row['kernel_sha256']==state['frozen_sources'][r['kernel_source']]
    else:assert all(r['implementation']=='delegate_qualified' for r in active) and len({r['prefix'] for r in active})==42
    result.update(mode=state['mode'],prefill_int8_record=row,prefill_precision_candidate=(state['mode']=='candidate'),same_NN_decode_control=True,
        old_BF16_prefill_coverage_condition_explicitly_adapted=True,other_qualified_checks_retained=True,new_model_Math_required=True)
    return result

# The actual mode below selects scheduling budget only. Both precision stacks
# intentionally use candidate NN, LM precision and prefill INT8 implementations.
def audit_runtime(root,state,official=False):
    import hashlib
    mode=state['mode'];budget={'control':4096,'candidate':8192}[mode]
    assert state['actual_budget']==budget and len(state['profile_environment'])==64
    assert state['profile_environment']['TIANSHU_DENSE_BUDGET_MODE']==mode
    result=audit_precision_runtime(root,dict(state,mode='candidate'),official=official)
    rows=[r for p in (root/'dense_budget_records').glob('budget_*.json') for r in read(p)]
    assert rows and any(r['default_applied'] for r in rows)
    for row in rows:
        assert row['mode']==mode and row['inherited_wrappers_called_once'] and row['RNG_and_request_data_not_accessed']
        assert row['original_arguments_and_return_value_forwarded'] and row['explicit_budget_preserved']
        for kind in ('native','policy'):
            path=row[kind+'_source'];assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==state['frozen_sources'][path]==row['native_source_sha256' if kind=='native' else 'policy_sha256']
        if row['default_applied']:assert row['before_budget'] is None and row['inherited_budget']==4096 and row['after_budget']==budget and row['max_num_seqs']==64
    routes=[read(p) for p in (root/'dense_budget_records').glob('hook_route_*.json')]
    assert routes and all(r['exact_one_hook_redirect'] and r['runpy_restored'] for r in routes)
    for route in routes:
        for path,digest in route['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==state['frozen_sources'][path]==digest
    guards=[read(p) for p in (root/'dense_budget_records').glob('guard_worker_*.json')];assert len(guards)==1
    guard=guards[0];path=guard['policy_source']
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==state['frozen_sources'][path]==guard['policy_sha256']
    assert path==str(BASE.parent/'tianshu_dense_budget8192_v2_20261021/long_policy.py') and guard['actual_budget']==budget and guard['actual_max_num_seqs']==64 and guard['actual_long_mode']=='candidate'
    result.update(mode=mode,scheduler_budget=budget,budget_default_records=rows,independent_long_guard_worker_binding=guard,
        actual_worker_and_scheduler_budget_verified=True,same_NN_LM_prefill_precision_in_both_arms=True,
        only_scheduling_budget_and_corresponding_startup_shapes_changed=True,Math_required_for_new_budget_profile=True)
    return result
