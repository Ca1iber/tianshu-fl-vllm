# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,math,statistics,sys
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
sys.path.insert(0,str(EXP/'tianshu_native_int8_down_official4k_v2_20261022'))
import quality_profile as q
ROOT=Path('/workspace/benchmark-results/20261022_native_int8_down_prefill_official4k_v2_tianshu')
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def main():
 assert not torch.cuda.is_initialized() and not (BASE/'OFFICIAL_GATE_32g.json').exists();state=q.read(ROOT/'result.json');assert state['phase']=='complete' and state['stage']=='official' and state['benchmark_exit_code']==0 and state['completed_runs']==4 and state['down_scope']=='prefill' and state['arm']=='candidate'
 for field,folder in (('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')):
  q.frozen_check(state[field])
  for p,h in state[field].items():assert q.digest(ROOT/folder/Path(p).relative_to('/'))==h
 assert len(state['frozen_sources'])==890 and state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=256,rounds=4,excluded_rounds=[1])
 independent=q.read(EXP/'tianshu_native_int8_down_prefill_quality_results_v2_20261022/QUALITY_GATE_32g.json');assert independent['complete'] and independent['new_model_accuracies_independently_recomputed'] and independent['C8_correct']==102 and independent['C64_correct']==103 and independent['model_frozen_sources']==state['frozen_sources']
 for p,h in independent['raw_files'].items():assert q.digest(p)==h
 for batch in (8,64):
  mathroot=Path(f'/workspace/benchmark-results/20261022_native_int8_down_prefill_math{batch}_v2_retry1_tianshu');mstate=q.read(mathroot/'result.json');q.frozen_check(mstate['controller_sources']);q.frozen_check(mstate['frozen_sources'])
 profile=q.audit_runtime(ROOT,state,official=True);assert profile['zero_preemptions'] and profile['actual_native_Down_runtime_banks']==42 and 4096 in profile['actual_native_Down_runtime_M_values'] and profile['actual_Down_M64_capture_banks']==0
 c8=q.read(Path(state['math_root'])/'accuracy_audit.json');comparison=q.audit_benchmark(ROOT,state,profile,c8,write=False);original=q.read(ROOT/'comparison.json')
 for key in ('total_tok_s','output_tok_s','ttft_ms','postfirst_spread_percent'):assert comparison[key]==original[key]
 rows=q.csv_rows(ROOT,'raw_runs_*.csv');summary=q.csv_rows(ROOT,'summary_*.csv');assert len(rows)==4 and len(summary)==1
 for row in rows:assert row['Run Status']=='SUCCESS' and float(row['Successful Requests'])==256 and [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256] and float(row['Total Input Tokens'])==256*4096 and float(row['Total Output Tokens'])==256*1024
 for key in ('Total tok/s','Output tok/s','Mean TTFT (ms)'):assert abs(float(summary[0][key])-statistics.mean(float(row[key]) for row in rows[1:]))<.011
 log=(ROOT/'server.log').read_text(errors='replace');assert 'Application startup complete.' in log;after=log.split('Application startup complete.',1)[1];assert not [line for line in after.splitlines() if 'Triton autotuning for function' in line]
 checkpoint=q.digest(MODEL);assert checkpoint==independent['checkpoint_sha256']=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c';assert 'No running processes found' in (ROOT/'hardware_after.txt').read_text()
 promote=(comparison['passes_public_4k_performance_gates'] and comparison['total_tok_s']>5777.22*1.005 and comparison['postfirst_spread_percent']<1 and comparison['ttft_ms']<=3459.44*1.01)
 (BASE/'CPU_COMPARISON_32g.json').write_text(json.dumps(comparison,indent=2)+'\n')
 with (BASE/'KEY_RUNS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,promotion_passed=promote,official_benchmark=True,scope='prefill',original13_serve_argv=True,environment_count=63,source_count=890,original877_sources_exact_subset=True,total_tok_s=comparison['total_tok_s'],output_tok_s=comparison['output_tok_s'],ttft_ms=comparison['ttft_ms'],hot_spread_percent=comparison['postfirst_spread_percent'],requests_per_round=256,runs=4,excluded_runs=[1],successful_requests=1024,post_ready_Triton_autotune_records=0,zero_preemptions_verified=True,all_inherited_runtime_and_new_Down_audits_recomputed=True,all_source_snapshots_verified=True,C8_correct=102,C64_correct=103,actual_native_Down_runtime_all42_M4096=True,M64_Down_BF16_retained=True,previous_formal_total_tok_s=5777.22,previous_formal_TTFT_ms=3459.44,vs_previous_formal_percent=100*(comparison['total_tok_s']/5777.22-1),vs_previous_TTFT_percent=100*(comparison['ttft_ms']/3459.44-1),historical_formal_comparison_not_same_run_AB=True,short_same_run_gain_percent=comparison['short_same_run_gain_percent'],checkpoint_postformal_sha256=checkpoint,model_frozen_sources=state['frozen_sources'],controller_sources=state['controller_sources'],server_command=state['server_command'],profile_environment=state['profile_environment'],additional_environment=state['additional_environment'],raw_root=str(ROOT),raw_result_sha256=q.digest(ROOT/'result.json'),raw_CSV_files={str(p):q.digest(p) for p in (ROOT/'benchmark_results').glob('*.csv')},independent_quality_gate_sha256=q.digest(EXP/'tianshu_native_int8_down_prefill_quality_results_v2_20261022/QUALITY_GATE_32g.json'),CPU_comparison_sha256=q.digest(BASE/'CPU_COMPARISON_32g.json'),verifier_source_sha256=q.digest(__file__),plan_sha256=q.digest(BASE/'PLAN_32g.md'),no_original_state_or_CSV_modified=True,full16K_not_run=True,full_competition_eligibility_not_asserted=True,adopted=False)
 assert not torch.cuda.is_initialized();(BASE/'OFFICIAL_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('model_frozen_sources','controller_sources','profile_environment','additional_environment','server_command','raw_CSV_files')}))
if __name__=='__main__':main()
