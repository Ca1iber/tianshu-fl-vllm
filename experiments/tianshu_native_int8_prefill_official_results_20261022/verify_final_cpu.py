# SPDX-License-Identifier: Apache-2.0
import json,sys,statistics,math,csv
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
sys.path.insert(0,str(EXP/'tianshu_native_int8_prefill_validation_v2_20261022'))
import quality_profile as q
ROOT=Path('/workspace/benchmark-results/20261022_native_int8_prefill_official4k_v2_tianshu')
REC=Path('/workspace/benchmark-results/20261022_native_int8_prefill_math8_CPU_recovery_tianshu')
C64=Path('/workspace/benchmark-results/20261022_native_int8_prefill_math64_v2_tianshu')
assert not torch.cuda.is_initialized() and not (BASE/'OFFICIAL_GATE_32g.json').exists()
state=q.read(ROOT/'result.json');assert state['phase']=='complete' and state['stage']=='official' and state['completed_runs']==4
assert 'No running processes found' in (ROOT/'hardware_after.txt').read_text()
for field,folder in (('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')):
    q.frozen_check(state[field])
    for p,h in state[field].items():assert q.digest(ROOT/folder/Path(p).relative_to('/'))==h
profile=q.audit_runtime(ROOT,state,official=True)
quality=q.read(REC/'accuracy_audit.json');quality64=q.read(C64/'accuracy_audit.json')
independent=q.read(EXP/'tianshu_native_int8_prefill_quality_results_20261022/QUALITY_GATE_32g.json')
assert independent['complete'] and independent['new_model_accuracies_independently_recomputed'] and independent['C8_correct']==independent['C64_correct']==101 and independent['C64_actual_native_prefill_banks']==42
assert independent['model_frozen_sources']==state['frozen_sources'] and len(state['frozen_sources'])==877
for p,h in independent['raw_results'].items():assert q.digest(p)==h
comparison=q.audit_benchmark(ROOT,state,profile,quality,write=False)
rows=q.csv_rows(ROOT,'raw_runs_*.csv');summary=q.csv_rows(ROOT,'summary_*.csv');assert len(rows)==4 and len(summary)==1
for row in rows:
    assert row['Run Status']=='SUCCESS' and int(float(row['Successful Requests']))==256
    assert [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
    assert int(float(row['Total Input Tokens']))==256*4096 and int(float(row['Total Output Tokens']))==256*1024
for key in ('Total tok/s','Output tok/s','Mean TTFT (ms)'):
    assert abs(float(summary[0][key])-statistics.mean(float(row[key]) for row in rows[1:]))<.011
values=[float(row['Total tok/s']) for row in rows[1:]];spread=100*(max(values)-min(values))/statistics.mean(values)
assert math.isclose(spread,comparison['postfirst_spread_percent'],rel_tol=1e-10)
log=(ROOT/'server.log').read_text(errors='replace');ready='Application startup complete.';assert ready in log
post=log.split(ready,1)[1];tuning=[line for line in post.splitlines() if 'Triton autotuning for function' in line];assert not tuning
checkpoint_line=(ROOT/'checkpoint_after.sha256').read_text().strip();checkpoint_hash=checkpoint_line.split()[0]
assert checkpoint_hash=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
screen=q.read(EXP/'tianshu_native_int8_prefill_results_v2_20261022/SCREEN_GATE_32g.json');assert screen['complete'] and screen['candidate_passes_perf_screen'] and screen['frozen_sources']==state['frozen_sources']
promote=comparison['passes_public_4k_performance_gates'] and comparison['total_tok_s']>5695.95 and spread<1
controller_sources=dict(state['controller_sources']);controller_sources.update({str(p):q.digest(p) for p in [BASE/'verify_final_cpu.py',BASE/'PLAN_32g.md',EXP/'tianshu_native_int8_prefill_quality_results_20261022/QUALITY_GATE_32g.json',EXP/'tianshu_native_int8_prefill_quality_results_20261022/verify_cpu.py',EXP/'tianshu_native_int8_prefill_results_v2_20261022/SCREEN_GATE_32g.json',EXP/'tianshu_native_int8_prefill_math8_recovery_20261022/CPU_RECOVERY_GATE_32g.json']});q.frozen_check(controller_sources)
assert not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,promotion_passed=promote,official_benchmark=True,original_serve_command=True,total_tok_s=comparison['total_tok_s'],output_tok_s=comparison['output_tok_s'],ttft_ms=comparison['ttft_ms'],postfirst_spread_percent=spread,successful_requests=1024,requests_per_run=256,runs=4,excluded_runs=[1],post_ready_Triton_autotune_records=0,full_runtime_audit_rechecked=True,all_source_snapshots_rechecked=True,fresh_official_C8_correct=101,C8_original_experiment_passed=False,C8_CPU_BF16_coverage_reconciliation=True,C8_native_prefill_bank_count=0,fresh_actual_C64_correct=101,C64_native_M4096_all42_verified=True,C64_M64_fused_graph_and_LM_INT8_verified=True,runtime_source_count=877,environment_count=63,original471_exact_subset=True,previous_formal_total_tok_s=5695.95,previous_formal_ttft_ms=3663.77,vs_previous_formal_percent=100*(comparison['total_tok_s']/5695.95-1),vs_previous_TTFT_percent=100*(comparison['ttft_ms']/3663.77-1),historical_comparison_is_not_same_run_AB=True,short_same_source_AB_percent=screen['throughput_improvement_percent'],public_4K_gates_passed=comparison['passes_public_4k_performance_gates'],public_baseline_total_tok_s=2028.01,public_baseline_TTFT_ms=11573.53,full_16K_not_rerun=True,full_competition_eligibility_not_asserted=True,checkpoint_after_sha256=checkpoint_hash,model_frozen_sources=state['frozen_sources'],server_command=state['server_command'],profile_environment=state['profile_environment'],additional_environment=state['additional_environment'],controller_sources=controller_sources,raw_root=str(ROOT),raw_result_sha256=q.digest(ROOT/'result.json'),raw_comparison_sha256=q.digest(ROOT/'comparison.json'),independent_quality_gate_sha256=q.digest(EXP/'tianshu_native_int8_prefill_quality_results_20261022/QUALITY_GATE_32g.json'),C8_quality_root=str(REC),C64_quality_root=str(C64),formal_raw_files={str(p.relative_to(ROOT)):q.digest(p) for p in [* (ROOT/'benchmark_results').glob('*.csv'),ROOT/'checkpoint_after.sha256']},verifier_source_sha256=q.digest(__file__))
(BASE/'OFFICIAL_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
with (BASE/'KEY_RUNS_32g.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in gate.items() if k not in ('model_frozen_sources','profile_environment','controller_sources','formal_raw_files','server_command')}))
