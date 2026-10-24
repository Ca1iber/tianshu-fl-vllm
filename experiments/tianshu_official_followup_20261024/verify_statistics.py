# SPDX-License-Identifier: Apache-2.0
"""Independently recompute all four official CSV rounds, not peak-only summaries."""
import csv,math,statistics
from pathlib import Path

def counter(path,name):
    rows=[float(v.rsplit(' ',1)[1]) for v in path.read_text().splitlines() if v.startswith('vllm:'+name+'{')]
    assert len(rows)==1,(name,len(rows));return rows[0]

def verify(root,state,read,digest):
    assert state['official'] and state['stage']=='official' and state['performance_preexcluded_rounds']==[1]
    case=state['benchmark_case'];assert case in ([4096,1024,64,256],[16384,1024,64,128])
    raw=list(root.glob('benchmark_results/raw_runs_*.csv'));summary=list(root.glob('benchmark_results/summary_*.csv'))
    assert len(raw)==len(summary)==1
    with raw[0].open(newline='') as f:rows=list(csv.DictReader(f))
    with summary[0].open(newline='') as f:averages=list(csv.DictReader(f))
    assert len(rows)==4 and len(averages)==1
    total=(case[0]+case[1])*case[3]
    for row in rows:
        assert [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==case
        assert row['Run Status']=='SUCCESS' and float(row['Successful Requests'])==case[3]
        assert float(row['Total Input Tokens'])==case[0]*case[3] and float(row['Total Output Tokens'])==case[1]*case[3]
        for k in ('Benchmark Duration (s)','Total tok/s','Output tok/s','Mean TTFT (ms)'):assert math.isfinite(float(row[k])) and float(row[k])>0
        assert math.isclose(float(row['Total tok/s']),total/float(row['Benchmark Duration (s)']),rel_tol=.0002)
    keys=('Total tok/s','Output tok/s','Mean TTFT (ms)')
    means={k:statistics.mean(float(r[k]) for r in rows[1:]) for k in keys}
    assert all(abs(means[k]-float(averages[0][k]))<=.011 for k in keys)
    spread=100*(max(float(r['Total tok/s']) for r in rows[1:])-min(float(r['Total tok/s']) for r in rows[1:]))/means['Total tok/s']
    delta={name:counter(root/'after_metrics.txt',name)-counter(root/'before_metrics.txt',name)
        for name in ('prefix_cache_queries_total','prefix_cache_hits_total','num_preemptions_total')}
    assert delta['prefix_cache_queries_total']>=4*case[0]*case[3] and 0<=delta['prefix_cache_hits_total']<=delta['prefix_cache_queries_total']
    assert delta['num_preemptions_total']>=0
    fraction=delta['prefix_cache_hits_total']/delta['prefix_cache_queries_total']
    result=dict(complete=True,official=True,case=case,rounds=4,excluded_rounds=[1],all_requests_succeeded=4*case[3],
        total_tok_s=means['Total tok/s'],output_tok_s=means['Output tok/s'],ttft_ms=means['Mean TTFT (ms)'],
        hot_spread_percent=spread,hot_spread_below1_percent=spread<1,prefix_hit_fraction=fraction,prefix_below1_percent=fraction<=.01,
        counters=delta,zero_preemptions=delta['num_preemptions_total']==0,
        raw_csv=str(raw[0]),raw_csv_sha256=digest(raw[0]),summary_csv=str(summary[0]),summary_csv_sha256=digest(summary[0]))
    return result
