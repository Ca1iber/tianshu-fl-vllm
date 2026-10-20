# SPDX-License-Identifier: Apache-2.0
"""Verify original 16K CSV and unchanged qualified server; zero preemption is not assumed."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from runtime_audit import audit_runtime
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261020_k8v16_official16k_tianshu')
LAUNCH=Path('/workspace/benchmark-results/20261020_k8v16_official16k_launcher_tianshu')
RELEASE=BASE.parent/'tianshu_k8v16_budget4096_release_20261019'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare():
    template=read(RELEASE/'serve_profile_template.json')
    profile=read(ROOT/'launcher_profile.json')
    expected={key:value.replace(template['old_output_root']+'/',str(ROOT)+'/') for key,value in template['environment'].items()}
    assert expected==profile['profile_environment'] and len(expected)==43
    assert profile['command']==template['server_command']
    assert profile['verified_source_count']==205 and profile['actual_default_budget']==4096
    timeout_pid=(LAUNCH/'server_pid.txt').read_text().strip()
    children=[]
    pending=[timeout_pid]
    while pending:
        parent=pending.pop()
        result=subprocess.run(['pgrep','-P',parent],capture_output=True,text=True)
        found=result.stdout.split()
        children.extend(found)
        pending.extend(found)
    matching=[]
    for pid in children:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        command=[piece.decode() for piece in command if piece]
        if len(command)>2 and command[2:] == profile['command'][1:]:
            matching.append((pid,command))
    assert len(matching)==1,matching
    pid,observed_command=matching[0]
    environment=dict(value.decode().split('=',1) for value in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in value)
    observed={key:environment.get(key) for key in expected}
    assert observed==expected
    (ROOT/'observed_server_environment.json').write_text(json.dumps(dict(server_pid=int(pid),environment=observed,command=profile['command'],actual_process_command=observed_command),indent=2)+'\n')
    sources=dict(template['frozen_sources'])
    for path in (Path(__file__),BASE/'runtime_audit.py',RELEASE/'serve_profile_template.json',BASE/'run.sh'):
        sources[str(path)]=sha(path)
    for path,expected_sha in sources.items():
        assert sha(path)==expected_sha,path
        destination=ROOT/'source_snapshot'/Path(path).relative_to('/')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(Path(path).read_bytes())
        assert sha(destination)==expected_sha
    for name in ('hardware_before.txt','hardware_after.txt'):
        target=ROOT/name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(LAUNCH/name)
    state=dict(output_root=str(ROOT),mode='candidate',actual_budget=4096,server_command=profile['command'],
               profile_environment=expected,frozen_sources=sources,phase='prepared_while_server_live',
               current_best_4k_tok_s=5101.45,quality_same_qualified_profile_102_of_105=True,
               profile_template_sha256=sha(RELEASE/'serve_profile_template.json'),
               numerical_format_and_sampler_not_changed=True,client_concurrency=64,requests_per_round=128)
    (ROOT/'verification_state.json').write_text(json.dumps(state,indent=2)+'\n')
    print('PREPARED',len(sources),'sources, observed server PID',pid,flush=True)

def complete():
    assert (LAUNCH/'ended.txt').exists() and (ROOT/'benchmark_ended.txt').exists()
    state=read(ROOT/'verification_state.json')
    for path,expected in state['frozen_sources'].items():
        assert sha(path)==sha(ROOT/'source_snapshot'/Path(path).relative_to('/'))==expected,path
    text=(ROOT/'benchmark.log').read_text()
    def csv_path(label):
        rows=[line[len(label):].strip() for line in text.splitlines() if line.startswith(label)]
        assert len(rows)==1,rows
        path=Path('/workspace')/rows[0]
        assert path.parent==Path('/workspace/benchmark_results') and path.is_file()
        return path
    raw,summary=csv_path('Raw CSV:'),csv_path('Summary CSV:')
    rows=list(csv.DictReader(raw.open(newline='')))
    averages=list(csv.DictReader(summary.open(newline='')))
    assert len(rows)==4 and len(averages)==1
    for row in rows:
        for key,value in (('Prefill',16384),('Decode',1024),('Conc',64),('Num Prompts',128),('Successful Requests',128),('Total Input Tokens',2097152),('Total Output Tokens',131072)):
            assert float(row[key])==value,(key,row[key])
        assert row['Run Status']=='SUCCESS'
        for key in ('Total tok/s','Output tok/s','Benchmark Duration (s)','Mean TTFT (ms)'):
            assert math.isfinite(float(row[key])) and float(row[key])>0
        duration=float(row['Benchmark Duration (s)'])
        assert abs(float(row['Total tok/s'])-2228224/duration)<.1
        assert abs(float(row['Output tok/s'])-131072/duration)<.02
    average=averages[0]
    for key in average:
        assert abs(float(average[key])-round(statistics.mean(float(row[key]) for row in rows[1:]),2))<.005001,(key,average[key])
    # This workload is memory-limited to about40 live sequences. All per-route
    # runtime checks remain; four-K-specific evidence of live B64 is not assumed.
    runtime=audit_runtime(ROOT,state,official=False)
    pure=[]
    for path in (ROOT/'topp_launch_records').glob('route_*.json'):
        row=read(path)
        if row.get('topp') and not row.get('topk'):
            pure.append(int(row['batch']))
    assert pure and max(pure)>=32
    preemptions=runtime['recorded_preemptions']
    assert preemptions['after_metrics.txt']>=preemptions['before_metrics.txt']
    throughputs=[float(row['Total tok/s']) for row in rows[1:]]
    mean=float(average['Total tok/s'])
    result=dict(complete=True,verified_utc=datetime.now(timezone.utc).isoformat(),official_16k=True,
                server_profile_same_as_qualified_4k=True,original_benchmark_sha256=sha('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'),
                test_case=[16384,1024,64,128],rounds=4,excluded_rounds=[1],successful_requests=512,failed_requests=0,
                total_tok_s=mean,output_tok_s=float(average['Output tok/s']),mean_TTFT_ms=float(average['Mean TTFT (ms)']),
                steady_spread_percent=100*(max(throughputs)-min(throughputs))/statistics.mean(throughputs),
                improvement_vs_public_baseline915_15_percent=100*(mean/915.15-1),
                historical_improvement_vs2665_42_percent=100*(mean/2665.42-1),
                historical_comparison_not_strict_single_change_AB=True,runtime_audit=runtime,
                max_recorded_native_TopP_batch=max(pure),client_concurrency64_does_not_require64_live_sequences=True,
                frozen_sources=state['frozen_sources'],profile_environment=state['profile_environment'],
                raw_CSV=dict(path=str(raw),sha256=sha(raw)),summary_CSV=dict(path=str(summary),sha256=sha(summary)),
                all_rows=rows,summary_row=average,raw_root=str(ROOT),
                same_profile_Math_Level3=dict(correct=102,total=105,accuracy_percent=102/105*100,fresh_Math_not_repeated_for_unchanged_server_profile=True))
    (ROOT/'OFFICIAL16K_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    (BASE/'OFFICIAL16K_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    (BASE/'RAW_RUNS_32g.csv').write_text(raw.read_text())
    (BASE/'SUMMARY_32g.csv').write_text(summary.read_text())
    print(json.dumps({key:result[key] for key in ('complete','total_tok_s','output_tok_s','mean_TTFT_ms','steady_spread_percent','historical_improvement_vs2665_42_percent')},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else complete()
