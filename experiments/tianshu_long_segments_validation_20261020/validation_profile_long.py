# SPDX-License-Identifier: Apache-2.0
"""Fresh quality and both official workloads bound to one measured long-segment profile."""
import importlib.util,json,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
SERVICE=BASE.parent/'tianshu_long_segments_service_20261020'
sys.path.insert(0,str(SERVICE))
p=SERVICE/'candidate_profile.py';s=importlib.util.spec_from_file_location('_long_segments_measured_profile',p)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
from validation_profile import BENCH_SHA,audit_math,csv_rows,normalized_environment,official_command
s=importlib.util.spec_from_file_location('_long_segments_quality_runtime',BASE/'audit_profile.py')
runtime=importlib.util.module_from_spec(s);s.loader.exec_module(runtime)
audit_runtime=runtime.audit_runtime

def prepare(root,screen_root):
    confirm=read(screen_root/'comparison.json')
    control=read(screen_root/'control/result.json');candidate=read(screen_root/'candidate/result.json')
    short=read('/workspace/benchmark-results/20261020_long_segments_service4k_tianshu/comparison.json')
    assert confirm['complete'] and confirm['input_tokens']==16384 and confirm['gain_percent']>=2
    assert short['complete'] and short['input_tokens']==4096 and short['gain_percent']>=-1
    assert control['phase']==candidate['phase']=='complete' and confirm['rounds_per_mode']==3 and confirm['excluded_rounds']==[1]
    assert min(r['total_tokens_per_second'] for r in candidate['screen_rounds'][1:])>max(r['total_tokens_per_second'] for r in control['screen_rounds'][1:])
    assert control['spread_percent']<1 and candidate['spread_percent']<1
    assert candidate['frozen_sources']==control['frozen_sources']==confirm['frozen_sources']==short['frozen_sources']
    frozen_check(confirm['frozen_sources'])
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert sources==confirm['frozen_sources'] and command==candidate['server_command']==control['server_command']
    expected={key:value.replace(candidate['output_root']+'/',str(root)+'/') for key,value in candidate['profile_environment'].items()}
    assert profile==expected and len(profile)==46 and profile['TIANSHU_LONG_SEGMENT_MODE']=='candidate'
    for path in (Path(__file__),BASE/'run_validation.py',BASE/'audit_profile.py',BASE/'batch_policy.json',BASE/'run_math_and_official.sh'):
        sources[str(path)]=digest(path)
    frozen_check(sources)
    for path,h in sources.items():
        destination=root/'source_snapshot'/Path(path).relative_to('/')
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(Path(path).read_bytes())
        assert digest(destination)==h
    return command,environment,profile,sources

def audit_benchmark(root,state,profile,quality,write=True):
    assert state['benchmark_exit_code']==0 and state['completed_runs']==8 and state['stage']=='official'
    assert quality['complete'] and quality['fresh_math_for_original_command_profile'] and not quality['reused_original_accuracy']
    assert quality['total']==105 and quality['accuracy']>.95
    assert state['server_command']==quality['server_command']==official_command()
    assert state['frozen_sources']==quality['frozen_sources']
    assert normalized_environment(state['profile_environment'],root)==normalized_environment(quality['profile_environment'],state['math_root'])
    rows=csv_rows(root,'raw_runs_*.csv');summary=csv_rows(root,'summary_*.csv')
    assert len(rows)==8 and len(summary)==2 and rows==state['raw_runs'] and summary==state['official_summaries']
    cases=[]
    for number,(length,count,baseline,bttft) in enumerate(((4096,256,2028.01,11573.53),(16384,128,915.15,599262.03))):
        runs=rows[number*4:(number+1)*4];average=summary[number]
        for row in runs:
            assert row['Run Status']=='SUCCESS' and float(row['Successful Requests'])==count
            assert [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[length,1024,64,count]
            assert float(row['Total Input Tokens'])==length*count and float(row['Total Output Tokens'])==1024*count
        for key in average:
            assert abs(float(average[key])-round(statistics.mean(float(r[key]) for r in runs[1:]),2))<.005001,(key,average[key])
        total=float(average['Total tok/s']);ttft=float(average['Mean TTFT (ms)'])
        values=[float(r['Total tok/s']) for r in runs[1:]]
        cases.append(dict(input_tokens=length,output_tokens=1024,concurrency=64,requests_per_round=count,
                          total_tok_s=total,output_tok_s=float(average['Output tok/s']),mean_TTFT_ms=ttft,
                          raw_runs=runs,summary=average,excluded_rounds=[1],successful_requests=count*4,
                          steady_spread_percent=100*(max(values)-min(values))/statistics.mean(values),
                          public_baseline_tok_s=baseline,improvement_vs_public_percent=100*(total/baseline-1),
                          passes_public_performance_gates=total>baseline*1.01 and ttft<=bttft*1.01))
    assert digest(root/'source_snapshot/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')==state['official_script_sha256']==BENCH_SHA
    result=dict(complete=True,official_benchmark=True,original_serve_command_verified=True,workloads=cases,
                successful_requests=1536,failed_requests=0,profile=profile,adopted=False,
                one_profile_for_both_workloads=True,fresh_math_correct=quality['correct'],fresh_math_total=105,
                fresh_math_accuracy=quality['accuracy'],frozen_sources=state['frozen_sources'],
                profile_environment=state['profile_environment'],server_command=state['server_command'],
                raw_response_scope='Original benchmark stdout and CSV; response bodies not persisted by original benchmark.')
    if write:(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
