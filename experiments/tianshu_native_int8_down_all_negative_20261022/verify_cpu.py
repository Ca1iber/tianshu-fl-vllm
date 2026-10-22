# SPDX-License-Identifier: Apache-2.0
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_native_int8_down_quality_20261022'))
import quality_profile as q
root=Path('/workspace/benchmark-results/20261022_native_int8_down_all_math64_tianshu');state=q.read(root/'result.json');assert state['phase']=='audit_failed' and state['evalscope_exit_code']==0 and state['down_scope']=='all' and state['quality_batch']==64
for field,folder in [('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')]:
 q.frozen_check(state[field])
 for p,h in state[field].items():assert q.digest(root/folder/Path(p).relative_to('/'))==h
own=q.down.audit(root,state);acc=q.legacy.original_math(root,state,{},write=False);assert acc['correct']==95 and acc['total']==105 and not acc['official_requirement_passed'];assert own['actual_native_Down_runtime_banks']==42 and own['actual_Down_M64_capture_banks']==42
rows=[q.read(p) for p in (root/'prefill_gemm_records').glob('mm_*.json')];shapes=sorted(r['shape'] for r in rows);assert shapes==[[2048,2048,2048],[2048,12288,2048]]
result=dict(complete=True,CPU_only=True,original_experiment_passed=False,original_phase='audit_failed',raw_evalscope_execution_complete=True,all105_successful=True,correct=95,total=105,accuracy=95/105,quality_requirement_passed=False,adopted=False,official_benchmark_not_run=True,C8_not_run=True,all_frozen_sources_and_snapshots_unchanged=True,model_source_count=len(state['frozen_sources']),controller_count=len(state['controller_sources']),actual_native_Down_runtime_banks=42,actual_native_Down_runtime_M_values=own['actual_native_Down_runtime_M_values'],actual_Down_M64_capture_banks=42,new_Down_memory_profile_all42_accounted=True,incorrect_indices=acc['incorrect_indices'],max_tokens_indices=acc['max_tokens_indices'],generation_stop_reasons=acc['generation_stop_reasons'],original_runtime_audit_failure='base_runtime_audit.py:100 expected old BF16 Down vendor-mm warm record; opaque fallback used eager F.linear, actual records only O/GateUp',original_full_runtime_audit_not_reconciled=True,actual_prefill_vendor_mm_warm_shapes=shapes,no_fictitious_old_Down_GEMM_record=True,raw_root=str(root),raw_result_sha256=q.digest(root/'result.json'),raw_accuracy_files=acc['raw_files'],verifier_sha256=q.digest(__file__))
(BASE/'NEGATIVE_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
