# SPDX-License-Identifier: Apache-2.0
"""Parse actual client output with the untouched competition CPU parser."""
import hashlib,importlib.util,json,math,re,sys
from pathlib import Path
BENCH=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(BENCH)=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
spec=importlib.util.spec_from_file_location('original_benchmark_parser',BENCH)
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
def counter(p,name):
    values=[float(line.rsplit(' ',1)[1]) for line in p.read_text().splitlines() if line.startswith('vllm:'+name+'{')]
    assert len(values)==1,(p,name);return values[0]
def main():
    root=Path(sys.argv[1]).resolve();i=int(sys.argv[2]);assert 1<=i<=4
    folder=root/'client_rounds';log=folder/f'round_{i}.log'
    before,after=folder/f'before_{i}.txt',folder/f'after_{i}.txt'
    config=json.loads((root/'client_configuration.json').read_text())
    assert config['case']==[4096,1024,64,128] and config['excluded_rounds']==[1]
    assert config['max_prefix_hit_fraction']==.01 and config['max_preemptions']==0
    seed=2026102400+i
    assert config['seeds'][i-1]==seed
    assert re.findall(r'\bseed=(\d+)',log.read_text())==[str(seed)]
    metrics=bench.extract_metrics(log.read_text())
    assert metrics['successful_requests']==128 and metrics['total_input_tokens']==524288 and metrics['total_output_tokens']==131072
    assert all(v is not None and math.isfinite(v) and v>=0 for v in metrics.values()),metrics
    assert metrics['benchmark_duration']>0
    assert math.isclose(metrics['total_token_throughput'],655360/metrics['benchmark_duration'],rel_tol=.0002)
    deltas={name:counter(after,name)-counter(before,name) for name in ('prefix_cache_queries_total','prefix_cache_hits_total','num_preemptions_total')}
    assert deltas['prefix_cache_queries_total']>=524288
    assert 0<=deltas['prefix_cache_hits_total']<=deltas['prefix_cache_queries_total']
    fraction=deltas['prefix_cache_hits_total']/deltas['prefix_cache_queries_total']
    result=dict(round=i,seed=seed,excluded=i==1,metrics=metrics,counters=deltas,prefix_hit_fraction=fraction,
        original_parser_sha256=digest(BENCH),raw_sha256={str(p):digest(p) for p in (log,before,after)})
    target=folder/f'round_{i}.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2)+'\n')
    assert fraction<=.01,('prefix_cache_gate',fraction)
    assert deltas['num_preemptions_total']==0,deltas
    row=bench.format_result(config['case'],metrics);row['Run Status']='SUCCESS'
    bench.append_csv(row,str(root/'benchmark_results/raw_runs_fresh_seed.csv'),bench.RAW_CSV_COLUMNS)
    if i==4:
        hot=[json.loads((folder/f'round_{j}.json').read_text())['metrics'] for j in (2,3,4)]
        summary=bench.format_result(config['case'],bench.average_metrics(hot),include_successful_requests=False)
        bench.append_csv(summary,str(root/'benchmark_results/summary_fresh_seed.csv'),bench.SUMMARY_CSV_COLUMNS)
    print(json.dumps({k:result[k] for k in ('round','seed','excluded','prefix_hit_fraction')}|{'total_tok_s':metrics['total_token_throughput'],'ttft_ms':metrics['mean_ttft_ms'],'successful':128}),flush=True)
if __name__=='__main__':main()
