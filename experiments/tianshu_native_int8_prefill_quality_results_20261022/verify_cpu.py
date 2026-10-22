# SPDX-License-Identifier: Apache-2.0
import json,sys
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
sys.path.insert(0,str(EXP/'tianshu_native_int8_prefill_validation_v2_20261022'))
import quality_profile as q
C8=Path('/workspace/benchmark-results/20261022_native_int8_prefill_math8_tianshu')
REC=Path('/workspace/benchmark-results/20261022_native_int8_prefill_math8_CPU_recovery_tianshu')
C64=Path('/workspace/benchmark-results/20261022_native_int8_prefill_math64_v2_tianshu')
assert not torch.cuda.is_initialized()
raw8=q.read(C8/'result.json');rec=q.read(REC/'result.json');quality8=q.read(REC/'accuracy_audit.json');raw64=q.read(C64/'result.json')
assert raw8['phase']=='audit_failed' and rec['phase']=='CPU_recovery_complete' and rec['original_experiment_passed'] is False
assert rec['original_result_sha256']==q.digest(C8/'result.json')==quality8['original_result_sha256']
assert raw64['phase']=='complete' and raw64['accuracy_passed'] and raw64['quality_batch']==64
for root,state in ((C8,raw8),(C64,raw64)):
    for field,subdir in (('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')):
        q.frozen_check(state[field])
        for p,h in state[field].items():assert q.digest(root/subdir/Path(p).relative_to('/'))==h
q.frozen_check(quality8['controller_sources'])
profiles={};accuracies={}
for batch,root,state in ((8,C8,raw8),(64,C64,raw64)):
    profile=q.audit_runtime(root,state,official=False);accuracy=q.original_math(root,state,profile,write=False)
    assert accuracy['complete'] and accuracy['total']==105 and accuracy['correct']==101 and accuracy['accuracy']>.95
    assert profile['verified_dense_batch']==batch and profile['LM_head_precision_in_verified_batch']==('BF16' if batch==8 else 'INT8')
    assert profile['all_inherited_attention_KV_scheduler_sampling_M8_M64_LM_checks_retained']
    profiles[batch]=profile;accuracies[batch]=accuracy
assert profiles[8]['actual_native_prefill_bank_count']==0 and not profiles[8]['actual_new_prefill_path_all_42_GateUp_verified'] and profiles[8]['actual_BF16_prefill_delegation_all42_verified']
assert profiles[64]['actual_native_prefill_bank_count']==42 and 4096 in profiles[64]['actual_native_prefill_M_values'] and profiles[64]['actual_C64_fused_replay_verified']
assert raw8['frozen_sources']==raw64['frozen_sources'] and len(raw64['frozen_sources'])==877
assert raw8['server_command']==raw64['server_command']==q.official_command()
a=q.normalized_environment(raw8['profile_environment'],C8);b=q.normalized_environment(raw64['profile_environment'],C64)
assert {k for k in a if a[k]!=b[k]}=={'TIANSHU_FUTURE_KV_MODE'} and a['TIANSHU_FUTURE_KV_MODE']=='candidate' and b['TIANSHU_FUTURE_KV_MODE']=='control'
assert raw8['additional_environment']==raw64['additional_environment'] and len(a)==62 and len({**a,**raw8['additional_environment']})==63
original=q.read(EXP/'tianshu_m_range_prefill_release_20261021/serve_profile_template.json')['frozen_sources'];assert len(original)==471 and all(raw64['frozen_sources'][k]==v for k,v in original.items())
assert not accuracies[64]['max_tokens_indices'] and accuracies[64]['generation_stop_reasons']=={'stop':105}
assert not torch.cuda.is_initialized()
report=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,new_model_accuracies_independently_recomputed=True,C8_correct=101,C64_correct=101,total_each=105,accuracy_each=101/105,all_210_requests_successful=True,interface_errors=0,C8_original_phase='audit_failed',C8_original_experiment_passed=False,C8_CPU_scope_reconciliation=True,C8_actual_native_prefill_banks=0,C8_actual_BF16_prefill_all42=True,C8_max_tokens_indices=accuracies[8]['max_tokens_indices'],C64_original_phase='complete',C64_actual_native_prefill_banks=42,C64_actual_native_M_values=profiles[64]['actual_native_prefill_M_values'],C64_actual_M64_fused_replay_verified=True,C64_LM_INT8_verified=True,C64_max_tokens_indices=[],C64_all_105_stop_naturally=True,source_count=877,original471_exact_subset=True,same_frozen_model_sources=True,same_original13_serve_argv=True,selected_environment_count=63,only_normalized_C64_environment_difference='TIANSHU_FUTURE_KV_MODE',all_inherited_checks_reexecuted=True,all_snapshotted_sources_rechecked=True,official_4K_result_not_yet_available=True,previous_qualified_formal_4K_total_tok_s=5695.95,model_frozen_sources=raw64['frozen_sources'],raw_results={str(C8/'result.json'):q.digest(C8/'result.json'),str(REC/'result.json'):q.digest(REC/'result.json'),str(REC/'accuracy_audit.json'):q.digest(REC/'accuracy_audit.json'),str(C64/'result.json'):q.digest(C64/'result.json'),str(C64/'accuracy_audit.json'):q.digest(C64/'accuracy_audit.json')},raw_accuracy_files={str(root):accuracies[batch]['raw_files'] for batch,root in ((8,C8),(64,C64))},verifier_source_sha256=q.digest(__file__))
(BASE/'QUALITY_GATE_32g.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('model_frozen_sources','raw_results','raw_accuracy_files')}))
