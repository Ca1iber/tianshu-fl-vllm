# SPDX-License-Identifier: Apache-2.0
"""Independently bind the untouched original N256 run to fresh quality/source evidence."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from candidate_profile import digest,read,frozen_check
from audit import audit_model,audit_math
from prerequisite_cpu import check
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261024_W4_QKVO64_fitted_official4k_tianshu')
BENCH=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')

def counter(path,name):
    values=[float(s.rsplit(' ',1)[1]) for s in path.read_text().splitlines()
        if s.startswith('vllm:'+name+'{')]
    assert len(values)==1,(path,name,values)
    return values[0]

def main():
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    assert not (BASE/'OFFICIAL_GATE_32g.json').exists()
    state=read(ROOT/'result.json')
    assert state['phase']=='complete' and state['evaluation_exit_code']==0
    assert state['mode']=='candidate' and state['stage']=='official' and state['official']
    assert state['benchmark_case']==[4096,1024,64,256] and state['performance_preexcluded_rounds']==[1]
    qualification=check()
    assert state['qualification']==qualification
    assert state['frozen_sources']==qualification['model_sources'] and len(state['frozen_sources'])==997
    assert state['server_command']==qualification['server_command'] and len(state['server_command'])==13
    assert state['evaluation_command']==['python3',str(BENCH),'--model','/workspace/MiniCPM5-2B',
        '--served-model-name','minicpm','--port','9031','--test-cases','[[4096,1024,64,256]]']
    frozen_check(state['controller_sources'])
    for path,sha in state['frozen_sources'].items():
        assert digest(Path(path))==sha==digest(ROOT/'source_snapshot'/Path(path).relative_to('/')),path
    full=state['full_environment']
    assert len(full)==65 and read(ROOT/'observed_environment.json')==full
    def normalized(env,root):return {k:v.replace(str(root),'<ROOT>') for k,v in env.items()}
    assert normalized(full,ROOT)==normalized(qualification['screen_candidate_full_environment'],qualification['screen_candidate_root'])
    profile=audit_model(ROOT,state,runtime=True)
    assert json.loads(json.dumps(profile))==read(ROOT/'profile_audit.json')
    assert profile['additional_payload_bytes']==210567168 and profile['QKVO64_capture_banks']==84
    assert profile['QKV64_BF16_capture_banks']==profile['O64_BF16_capture_banks']==0
    assert profile['actual_FULL64_replays']>=10 and profile['physical_KV_blocks']*16>=64*5120
    assert digest(BENCH)=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
    spec=importlib.util.spec_from_file_location('_untouched_original_official_parser',BENCH)
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    assert bench.RUNS==4 and bench.SKIP_FIRST==1
    raw_files=list((ROOT/'benchmark_results').glob('raw_runs_*.csv'))
    summaries=list((ROOT/'benchmark_results').glob('summary_*.csv'))
    assert len(raw_files)==len(summaries)==1
    with raw_files[0].open(newline='') as f:rows=list(csv.DictReader(f))
    with summaries[0].open(newline='') as f:summary=list(csv.DictReader(f))
    assert len(rows)==4 and len(summary)==1
    log=(ROOT/'evaluation.log').read_text()
    assert 'RUNS=4' in log and 'SKIP_FIRST=1' in log and 'TOTAL_CASES=1' in log
    chunks=re.split(r'^Running: 4096_1024_c64 \| Run ([1-4])/4\s*$',log,flags=re.MULTILINE)
    assert len(chunks)==9
    parsed=[];key_rows=[]
    for index,row in enumerate(rows,1):
        assert chunks[2*index-1]==str(index)
        text=chunks[2*index]
        metrics=bench.extract_metrics(text)
        assert metrics['successful_requests']==256 and metrics['total_input_tokens']==1048576 and metrics['total_output_tokens']==262144
        assert all(v is not None and math.isfinite(v) and v>=0 for v in metrics.values())
        assert metrics['benchmark_duration']>0
        assert math.isclose(metrics['total_token_throughput'],1310720/metrics['benchmark_duration'],rel_tol=.0002)
        assert re.findall(r'(?<![A-Za-z0-9_])seed=([0-9]+)',text)==['0']
        expected=bench.format_result([4096,1024,64,256],metrics)
        assert row['Run Status']=='SUCCESS'
        assert all(math.isclose(float(row[k]),float(v),abs_tol=.00001) for k,v in expected.items()),index
        parsed.append(metrics)
        key_rows.append(dict(round=index,excluded=index==1,total_tok_s=metrics['total_token_throughput'],
            output_tok_s=metrics['output_throughput'],ttft_ms=metrics['mean_ttft_ms'],
            benchmark_duration_s=metrics['benchmark_duration'],successful_requests=256,seed=0))
    expected_summary=bench.format_result([4096,1024,64,256],bench.average_metrics(parsed[1:]),include_successful_requests=False)
    assert all(math.isclose(float(summary[0][k]),float(v),abs_tol=.00001) for k,v in expected_summary.items())
    hot=parsed[1:]
    means={name:statistics.mean(r[name] for r in hot) for name in ('total_token_throughput','output_throughput','mean_ttft_ms')}
    spread=100*(max(r['total_token_throughput'] for r in hot)-min(r['total_token_throughput'] for r in hot))/means['total_token_throughput']
    before,after=ROOT/'before_metrics.txt',ROOT/'after_metrics.txt'
    delta={name:counter(after,name)-counter(before,name) for name in
        ('prefix_cache_queries_total','prefix_cache_hits_total','num_preemptions_total')}
    assert delta['prefix_cache_queries_total']>=4*1048576
    assert 0<=delta['prefix_cache_hits_total']<=delta['prefix_cache_queries_total']
    fraction=delta['prefix_cache_hits_total']/delta['prefix_cache_queries_total']
    passed=spread<1 and fraction<=.01 and delta['num_preemptions_total']==0
    raw_paths=[ROOT/name for name in ('result.json','profile_audit.json','evaluation.log','before_metrics.txt','after_metrics.txt','observed_environment.json')]
    raw_paths+=raw_files+summaries+[Path(__file__),Path(qualification['screen_gate_path'])]
    gate=dict(complete=True,CPU_only=True,official_case=[4096,1024,64,256],original_script_unchanged=True,
        all1024_requests_succeeded=True,all4_rounds_recorded=True,preexcluded_rounds=[1],
        hot_means=means,hot_spread_percent=spread,counters=delta,prefix_hit_fraction=fraction,
        quality=qualification['quality'],source_count=997,environment_count=65,
        additional_payload_bytes=210567168,physical_KV_blocks=profile['physical_KV_blocks'],
        actual_FULL64_replays=profile['actual_FULL64_replays'],fresh128_paired_gain_percent=qualification['paired_fresh128_change_percent'],
        stable_cache_preemption_gate=passed,historical_r4_total_tok_s=6050.94,
        historical_r4_change_percent=100*(means['total_token_throughput']/6050.94-1),
        ready_for_release_review=passed and means['total_token_throughput']>6050.94,
        adopted=False,raw_root=str(ROOT),raw_sha256={str(p):digest(p) for p in raw_paths})
    with (BASE/'OFFICIAL_KEY_RUNS_32g.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(key_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(key_rows)
    (BASE/'OFFICIAL_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k!='raw_sha256'},indent=2))
if __name__=='__main__':main()
