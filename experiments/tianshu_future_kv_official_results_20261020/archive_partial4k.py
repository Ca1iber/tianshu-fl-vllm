# SPDX-License-Identifier: Apache-2.0
"""Archive finished official 4K rows while the same original script runs16K."""
import csv,datetime,hashlib,io,json,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_future_kv_validation_20261020'))
import framework_profile as profile

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    root=Path(sys.argv[1]);math=Path(sys.argv[2]);state=read(root/'state.json');quality=read(math/'accuracy_audit.json')
    assert state['stage']=='official' and state['phase'] in ('benchmarking','evaluation_finished_pending_audit','complete')
    assert quality['complete'] and quality['total']==105 and quality['correct']==103 and quality['accuracy']>.95
    assert state['frozen_sources']==quality['frozen_sources'] and len(state['frozen_sources'])==261
    assert state['controller_sources']==quality['controller_sources'] and len(state['controller_sources'])==7
    profile.frozen_check(state['frozen_sources']);profile.frozen_check(state['controller_sources'])
    assert state['server_command']==quality['server_command']==profile.official_command()
    assert profile.normalized_environment(state['profile_environment'],root)==profile.normalized_environment(quality['profile_environment'],math)
    actual=read(root/'observed_server_environment.json')
    assert actual['command']==state['server_command'] and actual['environment']==state['profile_environment'] and len(actual['environment'])==52
    rawfiles=list((root/'benchmark_results').glob('raw_runs_*.csv'));summaries=list((root/'benchmark_results').glob('summary_*.csv'))
    assert len(rawfiles)==len(summaries)==1
    rawbytes=rawfiles[0].read_bytes();summarybytes=summaries[0].read_bytes()
    allrows=list(csv.DictReader(io.StringIO(rawbytes.decode())));allsummaries=list(csv.DictReader(io.StringIO(summarybytes.decode())))
    rows=[r for r in allrows if int(r['Prefill'])==4096];summary=[r for r in allsummaries if int(r['Prefill'])==4096]
    assert len(rows)==4 and len(summary)==1
    assert rows==[r for r in state['raw_runs'] if int(r['Prefill'])==4096]
    for r in rows:
        assert r['Run Status']=='SUCCESS' and float(r['Successful Requests'])==256
        assert [int(r[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
        assert float(r['Total Input Tokens'])==4096*256 and float(r['Total Output Tokens'])==1024*256
    means={key:statistics.mean(float(r[key]) for r in rows[1:]) for key in ('Total tok/s','Output tok/s','Mean TTFT (ms)')}
    for key,value in means.items():assert abs(float(summary[0][key])-value)<.011
    source=root/'source_snapshot/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'
    assert sha(source)==state['official_script_sha256']==profile.BENCH_SHA
    for name,values in (('RAW_4K_32g.csv',rows),('SUMMARY_4K_32g.csv',summary)):
        with (BASE/name).open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
    steady=[float(r['Total tok/s']) for r in rows[1:]]
    result=dict(complete_for_4K_only=True,full_official_dual_workload_not_yet_complete=True,adopted=False,
        official_script_unchanged=True,official_script_sha256=profile.BENCH_SHA,
        requests_successful_4K=1024,failed_requests_4K=0,rounds=4,excluded_rounds=[1],
        total_tok_s=float(summary[0]['Total tok/s']),output_tok_s=float(summary[0]['Output tok/s']),ttft_ms=float(summary[0]['Mean TTFT (ms)']),
        independently_recomputed_unrounded_means=means,steady_spread_percent=100*(max(steady)-min(steady))/statistics.mean(steady),
        prior_qualified_4K_total_tok_s=5099.65,vs_prior_4K_percent=100*(float(summary[0]['Total tok/s'])/5099.65-1),
        fresh_Math_correct=103,fresh_Math_total=105,fresh_Math_accuracy=103/105,
        same261_runtime_sources_52_environment_7_controllers_and_actual_original_argv_verified=True,
        final_runtime_audit_pending_until_full_16K_and_cleanup=True,raw_root=str(root),math_root=str(math),
        quality_audit_sha256=sha(math/'accuracy_audit.json'),raw_runs=rows,official_4K_summary=summary[0],
        raw_mutable_CSV_sha256_at_capture=hashlib.sha256(rawbytes).hexdigest(),
        summary_mutable_CSV_sha256_at_capture=hashlib.sha256(summarybytes).hexdigest(),
        source_CSVs_may_later_append16K=True,recorded_at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        verifier_source_sha256=sha(__file__))
    (BASE/'INTERIM_4K_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw_runs','official_4K_summary')}))
if __name__=='__main__':main()
