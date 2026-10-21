# SPDX-License-Identifier: Apache-2.0
"""Independent CPU reconciliation; original GPU experiment remains failed."""
import hashlib,json,os,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',VLLM_PLUGINS='',PYTHONPATH='')
BASE=Path(__file__).resolve().parent
RESUME=BASE.parent/'tianshu_m_range_prefill_resume_20261021'
sys.path.insert(0,str(RESUME))
import framework_profile as f
import torch
ROOT=f.MATH8;OUT=f.RECOVERY_RAW
assert not OUT.exists() and not torch.cuda.is_initialized();OUT.mkdir()
state=f.read(ROOT/'result.json');original_hash=f.digest(ROOT/'result.json')
assert state['phase']=='audit_failed' and state['evalscope_exit_code']==0
f.frozen_check(state['frozen_sources']);f.frozen_check(state['controller_sources'])
for name,kind in [('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')]:
 for path,h in state[name].items():assert f.digest(ROOT/kind/Path(path).relative_to('/'))==h
profile=f.audit_runtime(ROOT,state,official=False)
assert profile['verified_dense_batch']==8 and profile['actual_new_prefill_path_all_42_GateUp_verified']
assert not profile['actual_range_fused_all_42_layers_verified'] and profile['actual_fused_prefix_count']==0 and profile['actual_fused_M_values']==[]
prefill=profile['prefill_int8_record'];actual=[r for r in prefill['routes'].values() if r['stage']=='active' and not r['actually_inside_capture'] and r['implementation']=='prefill_A8W8']
assert {r['M'] for r in actual}=={256} and len({r['prefix'] for r in actual})==42
accuracy=f.audit_math(ROOT,state,profile,write=False)
assert accuracy['total']==105 and accuracy['correct']==103 and accuracy['official_requirement_passed']
accuracy.update(CPU_coverage_reconciliation=True,original_experiment_passed=False,original_experiment_phase='audit_failed',original_result_sha256=original_hash,original_math_output_root=str(ROOT),C8_actual_prefill_INT8_M_values=[256],C8_actual_range_fused_prefix_count=0,C64_supplementary_quality_required_before_official=True)
(OUT/'accuracy_audit.json').write_text(json.dumps(accuracy,indent=2)+'\n')
assert f.digest(ROOT/'result.json')==original_hash and not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,original_experiment_passed=False,original_experiment_phase='audit_failed',original_result_sha256=original_hash,original_raw_root=str(ROOT),recovery_raw_root=str(OUT),accuracy_audit_sha256=f.digest(OUT/'accuracy_audit.json'),correct=accuracy['correct'],total=105,accuracy=accuracy['accuracy'],official_requirement_passed=True,C8_actual_INT8_M_values=[256],C8_actual_range_fused_prefix_count=0,C8_does_not_cover_new_fusion=True,C64_actual_M4096_all42_quality_still_required=True,all_original_471_model_sources_and_snapshots_verified=True,all_original_controller_sources_and_snapshots_verified=True,original_controller_sources=state['controller_sources'],verifier_sha256=f.digest(__file__),verifier_source=str(Path(__file__).resolve()),raw_prediction_and_review_digests=accuracy['raw_files'])
(BASE/'MATH8_RECOVERY_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(OUT/'CPU_reconciliation.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k not in ('original_controller_sources','raw_prediction_and_review_digests')}))
