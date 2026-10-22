# SPDX-License-Identifier: Apache-2.0
"""Original smoke remains audit_failed; verify only its real integration scope."""
import importlib.util,json,hashlib,math
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261022_native_int8_prefill_service_v2_smoke_candidate_tianshu')
PATH=BASE.parent/'tianshu_native_int8_prefill_validation_20261022/native_structure.py'
spec=importlib.util.spec_from_file_location('_smoke_real_native_structure',PATH);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not torch.cuda.is_initialized()
state=read(ROOT/'result.json');assert state['phase']=='audit_failed' and state['error']=='AssertionError()'
assert state['workload']==dict(input_tokens=4096,output_tokens=128,concurrency=64,requests_per_round=64,rounds=1,excluded_rounds=[])
assert len(state['screen_rounds'])==1 and len(state['seeded_canaries'])==2 and len(state['int8_smoke_wave'])==8
for p,h in state['frozen_sources'].items():assert sha(p)==h==sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))
profile=audit.audit_runtime(ROOT,dict(state,quality_batch=64),official=False)
responses=state['seeded_canaries']+state['int8_smoke_wave']+state['screen_rounds'][0]['responses'];assert len(responses)==74
assert state['seeded_canaries'][0]['text']==state['seeded_canaries'][1]['text']
for row in responses:
    assert row['usage']['prompt_tokens']==row['input_tokens'] and row['usage']['completion_tokens']==row['output_tokens'] and row['usage']['total_tokens']==row['input_tokens']+row['output_tokens']
    assert row['text'] and 0<row['ttft_seconds']<=row['wall_seconds']
r=state['screen_rounds'][0];assert r['successful_requests']==len(r['responses'])==64 and math.isclose(r['total_tokens_per_second'],64*(4096+128)/r['duration_seconds'],rel_tol=1e-10)
assert not r['tuning_records']
native=profile['native_prefill_record'];routes=[r for r in native['routes'].values() if r['phase']=='runtime' and not r['capturing'] and r['implementation']=='native_A8W8_GateUp'];assert len({r['bank_id'] for r in routes if r['M']==4096})==42
fusion=read(next((ROOT/'nn_silu_fx_records').glob('route_*.json')));assert fusion['completed'] and fusion['all_42_M64_real_capture_routes_verified'] and fusion['warm_before_capture_RNG_unchanged']
assert len({r['prefix'] for r in fusion['routes'].values() if r['M']==64 and r['actually_inside_capture'] and r['component_binary_exact_match']})==42
assert 'No running processes found' in (ROOT/'hardware_after.txt').read_text()
assert not torch.cuda.is_initialized()
report=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,original_experiment_passed=False,original_phase=state['phase'],original_result_sha256=sha(ROOT/'result.json'),raw_root=str(ROOT),real_integration_scope_only=True,not_performance_or_Math_qualification=True,all_74_responses_successful=True,seeded_canaries_same=True,actual_M4096_native_all42=True,native_model_memory_profiler_and_actual_profile_peak_verified=True,existing_M64_capture_all42_verified=True,old23_full_binary_checks_retained=True,frozen_model_sources_unchanged=True,frozen_sources=state['frozen_sources'],all_inherited_throughput_audit_not_claimed=True,full_three_round_AB_still_required=True,post_ready_tuning_records_in_smoke_wave=0,verifier_source_sha256=sha(__file__),native_structure_source=str(PATH),native_structure_sha256=sha(PATH),reason='Original inherited baseline_pair_audit admission_checks>=194 assumes three rounds; smoke had 74 actual requests. Original audit_failed preserved.')
(BASE/'CPU_RECOVERY_GATE_32g.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='frozen_sources'}))
