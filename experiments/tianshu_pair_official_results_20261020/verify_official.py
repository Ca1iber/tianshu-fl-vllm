# SPDX-License-Identifier: Apache-2.0
"""Recompute original 4K CSV and stdout, fresh quality and actual paired-K bindings."""
import csv,hashlib,json,re,statistics,sys
from decimal import Decimal
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_k8v16_pair_validation_20261020'))
import framework_profile as profile

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    root=Path(sys.argv[1]);math=Path(sys.argv[2]);state=read(root/'result.json');ms=read(math/'result.json')
    assert state['phase']==ms['phase']=='complete'
    assert (root.parent/'20261020_pair_K_validation_pipeline_tianshu/complete.txt').is_file()
    assert state['stage']=='official' and ms['stage']=='math' and state['benchmark_exit_code']==0
    assert state['benchmark_command'][-1]=='[[4096,1024,64,256]]'
    profile.frozen_check(state['frozen_sources']);profile.frozen_check(state['controller_sources'])
    assert state['frozen_sources']==ms['frozen_sources'] and state['controller_sources']==ms['controller_sources']
    assert len(state['frozen_sources'])==282 and len(state['controller_sources'])==8 and len(state['profile_environment'])==54
    mruntime=profile.audit_runtime(math,ms,official=False)
    quality=profile.audit_math(math,ms,mruntime,write=False);assert quality==read(math/'accuracy_audit.json')
    assert quality['complete'] and quality['total']==105 and quality['accuracy']>.95
    runtime=profile.audit_runtime(root,state,official=True)
    assert runtime['complete'] and runtime['zero_preemptions'] and runtime['metadata_fusion_excluded']
    assert runtime['future_reservation']['tracked_requests']==runtime['future_reservation']['outstanding_blocks']==0
    assert runtime['verified_batch']==64 and runtime['pair_load_binding_verified'] and runtime['actual_expected_batch_capture_and_model_replay_verified']
    assert runtime['active_decode_source_sha256']=='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
    observed=read(root/'observed_server_environment.json')
    assert observed['environment']==state['profile_environment'] and observed['command']==state['server_command']==ms['server_command']==profile.official_command()
    assert profile.normalized_environment(state['profile_environment'],root)==profile.normalized_environment(ms['profile_environment'],math)
    current=profile.audit_benchmark(root,state,runtime,quality,write=False);assert current==read(root/'comparison.json')
    raw_files=list((root/'benchmark_results').glob('raw_runs_*.csv'));summary_files=list((root/'benchmark_results').glob('summary_*.csv'))
    assert len(raw_files)==len(summary_files)==1
    with raw_files[0].open(newline='') as file:raw=list(csv.DictReader(file))
    with summary_files[0].open(newline='') as file:summaries=list(csv.DictReader(file))
    assert len(raw)==4 and len(summaries)==1 and raw==state['raw_runs'] and summaries[0]==state['official_summary']
    for row in raw:
        assert row['Run Status']=='SUCCESS' and Decimal(row['Successful Requests'])==256
        assert [Decimal(row[key]) for key in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
        assert Decimal(row['Total Input Tokens'])==4096*256 and Decimal(row['Total Output Tokens'])==1024*256
        assert abs(Decimal(row['Total tok/s'])-Decimal(row['Output tok/s'])*5)<=Decimal('.03')
    columns={'Total tok/s':'total_tok_s','Output tok/s':'output_tok_s','Mean TTFT (ms)':'mean_TTFT_ms'}
    means={out:float(sum(Decimal(r[key]) for r in raw[1:])/3) for key,out in columns.items()}
    summary=summaries[0]
    assert all(abs(float(summary[key])-means[out])<=.0051 for key,out in columns.items())
    log=(root/'benchmark.log').read_text()
    for label,key in [('Total token throughput (tok/s)','Total tok/s'),('Output token throughput (tok/s)','Output tok/s'),('Mean TTFT (ms)','Mean TTFT (ms)')]:
        values=re.findall(r'^'+re.escape(label)+r':\s*([0-9.]+)',log,re.M)
        assert len(values)==4 and [Decimal(v) for v in values]==[Decimal(r[key]) for r in raw]
    assert sha(root/'source_snapshot/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')==state['official_script_sha256']==profile.BENCH_SHA
    prior_path=root.parent/'20261020_future_kv_official4k16k_tianshu/comparison.json'
    prior=read(prior_path);assert prior['complete'];old=next(r for r in prior['workloads'] if r['input_tokens']==4096)
    assert old['total_tok_s']==5103.41
    values=[float(r['Total tok/s']) for r in raw[1:]];spread=100*(max(values)-min(values))/statistics.mean(values)
    total=float(summary['Total tok/s']);ttft=float(summary['Mean TTFT (ms)']);gain=100*(total/old['total_tok_s']-1)
    adopted=bool(gain>=1 and ttft<=old['mean_TTFT_ms']*1.01 and spread<1 and current['passes_public_4k_performance_gates'])
    files=[root/'comparison.json',root/'result.json',root/'before_metrics.txt',root/'after_metrics.txt',root/'hardware_after.txt',math/'accuracy_audit.json',*raw_files,*summary_files]
    gate=dict(complete=True,independently_rechecked=True,adopted=adopted,original_comparison_not_overwritten=True,official_workloads=[dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=256,rounds=4,excluded_rounds=[1])],
        total_tok_s=total,output_tok_s=float(summary['Output tok/s']),mean_TTFT_ms=ttft,postfirst_spread_percent=spread,independent_raw_arithmetic_means=means,
        previous_formal_4K_total_tok_s=old['total_tok_s'],vs_previous_percent=gain,public_baseline_total_tok_s=2028.01,vs_public_percent=100*(total/2028.01-1),
        math_correct=quality['correct'],math_total=105,math_accuracy=quality['accuracy'],successful_requests=1024,failed_requests=0,
        runtime_source_count=282,environment_count=54,separate_controller_count=8,both_original_command_and_actual_process_environment_verified=True,
        active_decode_source_sha256=runtime['active_decode_source_sha256'],actual_B64_capture_and_replay_verified=True,preemptions=0,all_virtual_reservations_released=True,
        full_16K_not_run=True,prior_qualified_16K_from_different_profile=3885.91,full_competition_eligibility_not_asserted=True,
        official_script_sha256=profile.BENCH_SHA,raw_root=str(root),math_root=str(math),prior_qualified_raw_root=str(prior_path.parent),
        raw_files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},verifier_sha256=sha(__file__))
    (BASE/'OFFICIAL_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    (BASE/'RUNTIME_BINDING_GATE_32g.json').write_text(json.dumps(runtime,indent=2)+'\n')
    (root/'independent_official_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k!='raw_files'}))
if __name__=='__main__':main()
