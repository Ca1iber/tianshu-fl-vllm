# SPDX-License-Identifier: Apache-2.0
"""Offline original-CSV/provenance audit and compact evidence export; never starts a model."""
import argparse,csv,hashlib,json,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--official-root',required=True,type=Path);p.add_argument('--math-root',required=True,type=Path);args=p.parse_args()
root=args.official_root.resolve();math=args.math_root.resolve();base=Path(__file__).resolve().parent
read=lambda path:json.loads(Path(path).read_text())
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
state=read(root/'result.json');comparison=read(root/'comparison.json');quality=read(math/'accuracy_audit.json')
assert state['phase']=='complete' and comparison['complete'] and comparison['official_benchmark']
assert quality['complete'] and quality['total']==105 and quality['accuracy']>.95 and not quality['reused_original_accuracy']
assert read(math/'result.json')['phase']=='complete'
assert comparison['one_profile_for_both_workloads'] and comparison['successful_requests']==1536 and comparison['failed_requests']==0
assert state['frozen_sources']==quality['frozen_sources']==comparison['frozen_sources'] and len(state['frozen_sources'])==225
assert state['server_command']==quality['server_command']==comparison['server_command']
normalize=lambda env,path:{k:v.replace(str(path)+'/','<RUN>/') for k,v in env.items()}
assert normalize(state['profile_environment'],root)==normalize(quality['profile_environment'],math)
assert len(state['profile_environment'])==46
for directory in (root,math):
    observed=read(directory/'observed_server_environment.json')
    expected=state if directory==root else quality
    assert observed['command']==expected['server_command'] and observed['environment']==expected['profile_environment']
for name,digest in state['frozen_sources'].items():
    assert sha(name)==digest,name
    for directory in (root,math):assert sha(directory/'source_snapshot'/Path(name).relative_to('/'))==digest
assert state['official_script_sha256']=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
assert comparison['profile']['actual_FL_Graph_capture_and_replay_verified']
assert comparison['profile']['active_kernel_sha256']=='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
raw_paths=list((root/'benchmark_results').glob('raw_runs_*.csv'));summary_paths=list((root/'benchmark_results').glob('summary_*.csv'))
assert len(raw_paths)==len(summary_paths)==1
with raw_paths[0].open(newline='') as f:raw=list(csv.DictReader(f))
with summary_paths[0].open(newline='') as f:summaries=list(csv.DictReader(f))
assert raw==state['raw_runs'] and summaries==state['official_summaries'] and len(raw)==8 and len(summaries)==2
workloads=[]
for j,(length,count,old,baseline) in enumerate(((4096,256,5101.45,2028.01),(16384,128,3213.11,915.15))):
    runs=raw[j*4:(j+1)*4];summary=summaries[j]
    for row in runs:
        assert row['Run Status']=='SUCCESS' and int(float(row['Successful Requests']))==count
        assert [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[length,1024,64,count]
        assert float(row['Total Input Tokens'])==length*count and float(row['Total Output Tokens'])==1024*count
    for key,value in summary.items():
        expected=round(statistics.mean(float(row[key]) for row in runs[1:]),2)
        assert abs(float(value)-expected)<.005001,(key,value,expected)
    values=[float(row['Total tok/s']) for row in runs[1:]];total=float(summary['Total tok/s'])
    measured=comparison['workloads'][j]
    assert measured['input_tokens']==length and measured['total_tok_s']==total
    assert measured['excluded_rounds']==[1] and measured['successful_requests']==count*4
    workloads.append(dict(input_tokens=length,output_tokens=1024,concurrency=64,requests_per_round=count,
                          total_tok_s=total,output_tok_s=float(summary['Output tok/s']),TTFT_ms=float(summary['Mean TTFT (ms)']),
                          duration_s=float(summary['Benchmark Duration (s)']),excluded_rounds=[1],
                          spread_percent=100*(max(values)-min(values))/statistics.mean(values),
                          previous_qualified_total_tok_s=old,change_vs_previous_percent=100*(total/old-1),
                          previous_comparison_is_historical_not_paired_AB=True,
                          public_baseline_total_tok_s=baseline,change_vs_public_percent=100*(total/baseline-1)))
files={}
for source,target in ((raw_paths[0],base/'RAW_RUNS_32g.csv'),(summary_paths[0],base/'SUMMARY_32g.csv')):
    target.write_text(source.read_text().replace('\r\n','\n'))
    files[str(source)]=dict(original_sha256=sha(source),original_bytes=source.stat().st_size,
                            repository_file=target.name,repository_LF_sha256=sha(target))
server_log=(root/'server.log').read_text(errors='replace')
ready=server_log.find('Application startup complete');assert ready>=0
post_ready=[line for line in server_log[ready:].splitlines() if 'Triton autotuning for function' in line]
metrics={}
for name in ('before_metrics.txt','after_metrics.txt'):
    matches=re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$',(root/name).read_text(),re.MULTILINE)
    assert matches;metrics[name]=sum(map(float,matches))
assert 'No running processes found' in (root/'hardware_after.txt').read_text()
audit=dict(complete=True,official_benchmark=True,adopted=False,workloads=workloads,
           fresh_Math_correct=quality['correct'],fresh_Math_total=105,fresh_Math_accuracy=quality['accuracy'],
           same225_sources_46_environment_original_argv_verified=True,successful_requests=1536,failed_requests=0,
           actual_native_FL_Graph_verified=True,preemptions=metrics,
           post_ready_autotune_lines=post_ready,no_claim_of_zero_autotune=True,CSV_provenance=files,
           server_log_sha256=sha(root/'server.log'),raw_comparison_sha256=sha(root/'comparison.json'),
           archive_script_sha256=sha(Path(__file__)),original_roots=dict(official=str(root),math=str(math)))
(base/'OFFICIAL_AUDIT_32g.json').write_text(json.dumps(audit,indent=2)+'\n')
(base/'OFFICIAL_GATE_32g.json').write_bytes((root/'comparison.json').read_bytes())
print(json.dumps({k:v for k,v in audit.items() if k not in ('post_ready_autotune_lines','CSV_provenance')}))
