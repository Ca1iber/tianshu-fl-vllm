# SPDX-License-Identifier: Apache-2.0
"""Read-only independent recomputation; never starts or imports a GPU runtime."""
import csv,importlib.util,json,math,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
spec=importlib.util.spec_from_file_location('_LM64_independent_official',EXP/'tianshu_W4_LMpad_official4k_v2_20261023/quality_profile.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
ROOT=Path('/workspace/benchmark-results/20261023_W4_LMpad_v2_official4k_tianshu')
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def main():
 assert not (BASE/'OFFICIAL_GATE_32g.json').exists()
 state=q.read(ROOT/'result.json');assert state['phase']=='complete' and state['stage']=='official' and state['benchmark_exit_code']==0 and state['completed_runs']==4 and state['arm']=='candidate'
 assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=256,rounds=4,excluded_rounds=[1]) and len(state['server_command'])==13
 for field,folder in (('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')):
  q.frozen_check(state[field])
  for p,h in state[field].items():assert q.digest(ROOT/folder/Path(p).relative_to('/'))==h
 assert len(state['frozen_sources'])==944
 quality_path=EXP/'tianshu_W4_LMpad_quality_results_v2_20261023/QUALITY_GATE_32g.json';quality=q.read(quality_path)
 assert quality['complete'] and quality['model_quality_qualified'] and quality['C8_correct']>=100 and quality['C64_correct']>=100 and quality['fresh_entire_batch_8_LM_precision_changed'] and quality['model_frozen_sources']==state['frozen_sources']
 for p,h in quality['raw_files'].items():assert q.digest(p)==h
 profile=q.audit_runtime(ROOT,state,official=True)
 assert profile['zero_preemptions'] and profile['actual_Down64_W4_capture_banks']==42 and profile['old_Down64_BF16_capture_banks']==0 and profile['new_W4_all42_capture_linked_to_actual_B64_model_replays']
 assert profile['new_LM64_W4_actual_runtime_calls']>=100 and profile['old_LM64_W8_calls_absent'] and profile['new_LM64_load_and_native_peak_accounting_verified'];assert profile['new_LM8_W4_actual_runtime_calls']>=1 and profile['old_LM1_64_BF16_W8_calls_absent']
 c8=q.read(Path(state['math_root'])/'accuracy_audit.json');comparison=q.audit_benchmark(ROOT,state,profile,c8,write=False)
 assert comparison==q.read(ROOT/'comparison.json')
 rows=q.csv_rows(ROOT,'raw_runs_*.csv');summary=q.csv_rows(ROOT,'summary_*.csv');assert len(rows)==4 and len(summary)==1
 for row in rows:
  assert row['Run Status']=='SUCCESS' and int(float(row['Successful Requests']))==256
  assert [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
  assert float(row['Total Input Tokens'])==256*4096 and float(row['Total Output Tokens'])==256*1024
 for key in ('Total tok/s','Output tok/s','Mean TTFT (ms)'):assert abs(float(summary[0][key])-statistics.mean(float(row[key]) for row in rows[1:]))<.011
 log=(ROOT/'server.log').read_text(errors='replace');assert 'Application startup complete.' in log
 assert not any('Triton autotuning for function' in line for line in log.split('Application startup complete.',1)[1].splitlines())
 assert 'No running processes found' in (ROOT/'hardware_after.txt').read_text()
 checkpoint=q.digest(MODEL);assert checkpoint==quality['checkpoint_sha256']=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
 published=q.read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');assert len(published['frozen_sources'])==903 and all(state['frozen_sources'][p]==h for p,h in published['frozen_sources'].items())
 with (BASE/'KEY_RUNS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 (BASE/'CPU_COMPARISON_32g.json').write_text(json.dumps(comparison,indent=2)+'\n')
 gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,official_benchmark=True,promotion_passed=comparison['candidate_passes_promotion_performance_gates'],adopted=False,total_tok_s=comparison['total_tok_s'],output_tok_s=comparison['output_tok_s'],ttft_ms=comparison['ttft_ms'],hot_spread_percent=comparison['postfirst_spread_percent'],previous_formal_total_tok_s=5950.75,previous_formal_TTFT_ms=3174.58,vs_previous_formal_percent=comparison['vs_previous_formal_percent'],historical_formal_comparison_not_same_run_AB=True,short_same_run_gain_percent=comparison['short_same_run_gain_percent'],requests_per_round=256,runs=4,excluded_runs=[1],successful_requests=1024,post_ready_Triton_autotune_records=0,zero_preemptions_verified=True,all_inherited_and_new_runtime_audits_recomputed=True,new_W4_M64_all42_captures_linked_to_actual_model_replays=True,old_Down_M64_BF16_captures=0,QKV64_O64_BF16_retained=True,LM8_W4_verified=True,LM64_native_W4_verified=True,C8_and_C64_fresh_normal_runs_verified=True,C8_correct=quality['C8_correct'],C64_correct=quality['C64_correct'],total_per_batch=105,model_source_count=944,original903_sources_exact_subset=True,original13_serve_argv=True,environment_count=63,checkpoint_sha256=checkpoint,model_frozen_sources=state['frozen_sources'],controller_sources=state['controller_sources'],server_command=state['server_command'],profile_environment=state['profile_environment'],additional_environment=state['additional_environment'],raw_root=str(ROOT),raw_result_sha256=q.digest(ROOT/'result.json'),raw_CSV_files={str(p):q.digest(p) for p in (ROOT/'benchmark_results').glob('*.csv')},independent_quality_gate_sha256=q.digest(quality_path),CPU_comparison_sha256=q.digest(BASE/'CPU_COMPARISON_32g.json'),key_runs_sha256=q.digest(BASE/'KEY_RUNS_32g.csv'),verifier_source_sha256=q.digest(__file__),plan_sha256=q.digest(BASE/'PLAN_32g.md'),no_original_raw_modified=True,full16K_not_run=True,full_competition_eligibility_not_asserted=True)
 assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
 (BASE/'OFFICIAL_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('model_frozen_sources','controller_sources','server_command','profile_environment','additional_environment','raw_CSV_files')}))
if __name__=='__main__':main()
