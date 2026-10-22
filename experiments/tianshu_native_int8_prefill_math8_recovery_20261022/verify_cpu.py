# SPDX-License-Identifier: Apache-2.0
"""Fresh C8 raw is valid for its actual BF16-prefill scope; never fake INT8 hits."""
import json,sys
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
VALIDATION=BASE.parent/'tianshu_native_int8_prefill_validation_v2_20261022';sys.path.insert(0,str(VALIDATION))
import quality_profile as q
root=Path('/workspace/benchmark-results/20261022_native_int8_prefill_math8_tianshu')
out=Path('/workspace/benchmark-results/20261022_native_int8_prefill_math8_CPU_recovery_tianshu');assert not out.exists();out.mkdir()
assert not torch.cuda.is_initialized();state=q.read(root/'result.json');assert state['phase']=='audit_failed' and state['quality_batch']==8 and state['evalscope_exit_code']==0
q.frozen_check(state['frozen_sources']);q.frozen_check(state['controller_sources']);assert len(state['frozen_sources'])==877
profile=q.audit_runtime(root,state,official=False)
assert profile['verified_dense_batch']==8 and profile['LM_head_precision_in_verified_batch']=='BF16' and profile['actual_native_prefill_bank_count']==0 and profile['actual_native_prefill_M_values']==[]
assert not profile['actual_new_prefill_path_all_42_GateUp_verified'] and profile['actual_BF16_prefill_delegation_all42_verified'] and profile['actual_prefill_INT8_shapes']==[]
assert profile['all_inherited_attention_KV_scheduler_sampling_M8_M64_LM_checks_retained']
accuracy=q.audit_math(root,state,profile,write=False);assert accuracy['complete'] and accuracy['correct']==101 and accuracy['total']==105 and accuracy['accuracy']>.95
controllers=dict(state['controller_sources']);controllers.update({str(p):q.digest(p) for p in [*VALIDATION.glob('*.py'),VALIDATION/'PLAN_32g.md',BASE/'verify_cpu.py',BASE/'PLAN_32g.md']})
accuracy.update(CPU_coverage_reconciliation=True,original_experiment_passed=False,actual_math_raw_root=str(root),original_result_sha256=q.digest(root/'result.json'),controller_sources=controllers,actual_new_prefill_precision_coverage=False,actual_BF16_prefill_delegation_all42_verified=True,actual_BF16_prefill_M_values=profile['actual_BF16_prefill_M_values'],C64_native_INT8_all42_fresh_quality_required=True,previous_model_accuracy_not_reused=True,CPU_verifier=str(BASE/'verify_cpu.py'),CPU_verifier_sha256=q.digest(__file__))
(out/'accuracy_audit.json').write_text(json.dumps(accuracy,indent=2)+'\n')
result=dict(phase='CPU_recovery_complete',complete=True,CPU_only=True,CUDA_not_initialized=True,original_experiment_passed=False,original_phase='audit_failed',original_result_sha256=q.digest(root/'result.json'),actual_math_raw_root=str(root),accuracy=accuracy['accuracy'],correct=101,total=105,actual_C8_decode_NN_and_LM_BF16_verified=True,all_actual_prefill_BF16_fallback=True,native_runtime_bank_count=0,actual_prefill_precision_coverage=False,full_inherited_checks_retained=True,all_105_predictions_reviews_rechecked=True,model_frozen_sources=state['frozen_sources'],controller_sources=controllers,accuracy_audit_sha256=q.digest(out/'accuracy_audit.json'),raw_recovery_root=str(out))
q.frozen_check(controllers);assert not torch.cuda.is_initialized();(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(BASE/'CPU_RECOVERY_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
for p,h in controllers.items():
    target=out/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert q.digest(target)==h
print(json.dumps({k:v for k,v in result.items() if k not in ('model_frozen_sources','controller_sources')}))
