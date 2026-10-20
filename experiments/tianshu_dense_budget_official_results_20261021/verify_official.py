# SPDX-License-Identifier: Apache-2.0
"""Original4K CSV/stdout plus fresh C8 and C64 W8A8 and bank identity."""
import csv,hashlib,json,re,statistics,sys
from decimal import Decimal
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_dense_budget_validation_v2_20261021'))
import framework_profile as profile
sys.path.insert(0,str(BASE.parent/'tianshu_dense_budget_audit_recovery_20261021'))
from recover import checked_audit
profile.audit_runtime=checked_audit().audit_runtime

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def banks(runtime):return sorted([{k:r[k] for k in ('role','prefix','shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes')} for r in runtime['dense_int8_record']['banks']],key=lambda r:r['prefix'])
def main():
    root=Path(sys.argv[1]);math=Path(sys.argv[2]);state=read(root/'result.json');ms=read(math/'rechecked_result.json')
    assert state['phase']==ms['phase']=='complete' and state['stage']=='official' and ms['stage']=='math'
    assert (root.parent/'20261021_dense_budget_v2_official_recovery_runner_tianshu/complete.txt').is_file()
    assert state['benchmark_exit_code']==0 and state['benchmark_command'][-1]=='[[4096,1024,64,256]]'
    recovery=read(root/'supplemental_controller_sources.json');assert recovery['GPU_Math_not_rerun'] and recovery['primary_controller_sources']==state['controller_sources']
    for path,digest in recovery['supplemental_controller_sources'].items():assert sha(path)==sha(root/'supplemental_source_snapshot'/Path(path).relative_to('/'))==digest
    assert state['frozen_sources']==ms['frozen_sources'] and state['controller_sources']==ms['controller_sources']
    profile.frozen_check(state['frozen_sources']);profile.frozen_check(state['controller_sources'])
    assert len(state['frozen_sources'])==374 and len(state['controller_sources'])==9 and len(state['profile_environment'])==64
    assert state['profile_environment']['VLLM_DISABLE_COMPILE_CACHE']=='1'
    assert state['actual_budget']==ms['actual_budget']==8192 and state['profile_environment']['TIANSHU_DENSE_BUDGET_MODE']=='candidate'
    mruntime=profile.audit_runtime(math,ms,official=False);quality=profile.audit_math(math,ms,mruntime,write=False)
    assert quality==read(math/'rechecked_accuracy_audit.json') and quality['complete'] and quality['total']==105 and quality['accuracy']>.95
    assert quality['actual_C8_GateUp_integer_path_verified'] and quality['actual_C8_LM_BF16_path_verified'] and mruntime['verified_dense_batch']==8
    runtime=profile.audit_runtime(root,state,official=True)
    assert runtime['verified_dense_batch']==64 and runtime['actual_42_GateUp_banks_capture_and_replay_verified'] and runtime['actual_LM_head_outside_model_Graph_calls_verified']
    assert runtime['scheduler_budget']==mruntime['scheduler_budget']==8192 and runtime['warm_points']==mruntime['warm_points']==99
    assert runtime['exact_one_long_hook_redirect_verified'] and mruntime['exact_one_long_hook_redirect_verified']
    assert runtime['zero_preemptions'] and runtime['future_reservation']['tracked_requests']==runtime['future_reservation']['outstanding_blocks']==0
    signature=banks(runtime);assert len(signature)==43 and signature==banks(mruntime)
    screen=read(Path(state['screen_root'])/'SCREEN_GATE_32g.json');screened=read(Path(screen['raw_roots']['candidate'])/'result.json');assert signature==banks(screened['profile_audit'])
    observed=read(root/'observed_server_environment.json');assert observed['environment']==state['profile_environment']
    assert observed['command']==state['server_command']==ms['server_command']==profile.official_command()
    assert profile.normalized_environment(state['profile_environment'],root)==profile.normalized_environment(ms['profile_environment'],math)
    current=profile.audit_benchmark(root,state,runtime,quality,write=False);assert current==read(root/'comparison.json')
    raw_files=list((root/'benchmark_results').glob('raw_runs_*.csv'));summary_files=list((root/'benchmark_results').glob('summary_*.csv'));assert len(raw_files)==len(summary_files)==1
    with raw_files[0].open(newline='') as file:raw=list(csv.DictReader(file))
    with summary_files[0].open(newline='') as file:summaries=list(csv.DictReader(file))
    assert len(raw)==4 and len(summaries)==1 and raw==state['raw_runs'] and summaries[0]==state['official_summary']
    for row in raw:
        assert row['Run Status']=='SUCCESS' and Decimal(row['Successful Requests'])==256
        assert [Decimal(row[key]) for key in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
        assert Decimal(row['Total Input Tokens'])==4096*256 and Decimal(row['Total Output Tokens'])==1024*256
        assert abs(Decimal(row['Total tok/s'])-Decimal(row['Output tok/s'])*5)<=Decimal('.03')
    columns={'Total tok/s':'total_tok_s','Output tok/s':'output_tok_s','Mean TTFT (ms)':'mean_TTFT_ms'}
    means={out:float(sum(Decimal(r[key]) for r in raw[1:])/3) for key,out in columns.items()};summary=summaries[0]
    assert all(abs(float(summary[key])-means[out])<=.0051 for key,out in columns.items())
    log=(root/'benchmark.log').read_text()
    for label,key in [('Total token throughput (tok/s)','Total tok/s'),('Output token throughput (tok/s)','Output tok/s'),('Mean TTFT (ms)','Mean TTFT (ms)')]:
        values=re.findall(r'^'+re.escape(label)+r':\s*([0-9.]+)',log,re.M);assert len(values)==4 and [Decimal(v) for v in values]==[Decimal(r[key]) for r in raw]
    assert sha(root/'source_snapshot/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')==state['official_script_sha256']==profile.BENCH_SHA
    prior_path=BASE.parent/'tianshu_lm_precision_official_results_20261021/OFFICIAL_GATE_32g.json';prior=read(prior_path)
    assert prior['complete'] and prior['total_tok_s']==5482.79
    total=float(summary['Total tok/s']);ttft=float(summary['Mean TTFT (ms)']);values=[float(r['Total tok/s']) for r in raw[1:]]
    spread=100*(max(values)-min(values))/statistics.mean(values);gain=100*(total/prior['total_tok_s']-1)
    qualifies_for_best_review=bool(gain>0 and ttft<=prior['mean_TTFT_ms']*1.01 and spread<1 and current['passes_public_4k_performance_gates']);adopted=False
    files=[root/'comparison.json',root/'result.json',root/'before_metrics.txt',root/'after_metrics.txt',root/'hardware_after.txt',math/'rechecked_accuracy_audit.json',math/'rechecked_result.json',math/'postprocessing_recovery.json',math/'result.json',*raw_files,*summary_files]
    gate=dict(complete=True,independently_rechecked=True,adopted=adopted,original_comparison_not_overwritten=True,official_workloads=[dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=256,rounds=4,excluded_rounds=[1])],
        original_Math_audit_failure_retained=True,postprocessing_only_recovery=True,supplemental_Python_controller_source_count=2,
        qualifies_for_best_review=qualifies_for_best_review,adoption_not_performed_by_archiver=True,actual_budget=8192,actual_warm_points=99,
        total_tok_s=total,output_tok_s=float(summary['Output tok/s']),mean_TTFT_ms=ttft,postfirst_spread_percent=spread,independent_raw_arithmetic_means=means,
        previous_formal_4K_total_tok_s=5482.79,vs_previous_percent=gain,public_baseline_total_tok_s=2028.01,vs_public_percent=100*(total/2028.01-1),
        math_correct=quality['correct'],math_total=105,math_accuracy=quality['accuracy'],successful_requests=1024,failed_requests=0,
        runtime_source_count=374,environment_count=64,separate_controller_count=9,compile_cache_disabled=True,both_original_argv_and_actual_environment_verified=True,
        actual_42_GateUp_banks_capture_and_replay_C8_C64_verified=True,actual_LM_head_C8_BF16_and_C64_INT8_verified=True,actual_new_prefill_all_42_GateUp_verified=True,all_43_packed_banks_same_in_screen_Math_and_official=True,
        extra_packed_weight_bytes=1326938112,actual_KV_blocks=runtime['actual_KV_blocks'],preemptions=0,all_virtual_reservations_released=True,
        full_16K_not_run=True,prior_16K_3885_91_is_a_different_profile=True,full_competition_eligibility_not_asserted=True,
        official_script_sha256=profile.BENCH_SHA,raw_root=str(root),math_root=str(math),prior_qualified_raw_root=prior['raw_root'],
        raw_files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},verifier_sha256=sha(__file__))
    (BASE/'OFFICIAL_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    runtime_compact=dict(complete=True,frozen_sources=state["frozen_sources"],controller_sources=state["controller_sources"],profile_environment=state["profile_environment"],actual_KV_blocks=runtime["actual_KV_blocks"],verified_dense_batch=runtime["verified_dense_batch"],LM_head_precision_in_verified_batch=runtime["LM_head_precision_in_verified_batch"],actual_42_GateUp_banks_capture_and_replay_verified=runtime["actual_42_GateUp_banks_capture_and_replay_verified"],actual_LM_head_outside_model_Graph_calls_verified=runtime["actual_LM_head_outside_model_Graph_calls_verified"],actual_new_prefill_path_all_42_GateUp_verified=runtime["actual_new_prefill_path_all_42_GateUp_verified"],zero_preemptions=runtime["zero_preemptions"],outstanding_virtual_reservations=runtime["future_reservation"]["outstanding_blocks"])
    (BASE/'RUNTIME_BINDING_GATE_32g.json').write_text(json.dumps(runtime_compact,indent=2)+'\n')
    (BASE/'PACKED_WEIGHT_SIGNATURE_32g.json').write_text(json.dumps(signature,indent=2)+'\n')
    (root/'independent_official_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k!='raw_files'}))
if __name__=='__main__':main()
