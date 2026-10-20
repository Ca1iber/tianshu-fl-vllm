# SPDX-License-Identifier: Apache-2.0
"""Final paired-K profile: identical 282 runtime sources to the successful screen."""
import importlib.util,json,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_original_command_20261016'))
from validation_profile import BENCH_SHA,audit_math as original_math,csv_rows,normalized_environment,official_command

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
parent=load('_pair_final_service_profile',BASE.parent/'tianshu_k8v16_pair_service_20261020/candidate_profile.py')
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
runtime=load('_pair_final_runtime',BASE/'audit_profile.py');audit_runtime=runtime.audit_runtime

def prepare(root,screen_root):
    gate_path=BASE.parent/'tianshu_k8v16_pair_service_20261020/DIAGNOSTIC_4K_GATE_32g.json'
    gate=read(gate_path);control=read(screen_root/'control/result.json');candidate=read(screen_root/'candidate/result.json')
    assert gate['complete'] and gate['independently_rechecked'] and gate['gain_percent']>=2
    assert gate['input_tokens']==4096 and gate['requests_per_round']==64 and gate['excluded_rounds']==[1]
    assert gate['frozen_source_count']==282 and gate['profile_environment_count']==54
    assert control['phase']==candidate['phase']=='complete'
    assert control['frozen_sources']==candidate['frozen_sources']==gate['frozen_sources']
    assert min(r['total_tokens_per_second'] for r in candidate['screen_rounds'][1:])>max(r['total_tokens_per_second'] for r in control['screen_rounds'][1:])
    assert control['spread_percent']<1 and candidate['spread_percent']<1
    word_path=BASE.parent/'tianshu_k8v16_word_decode_20261020/GPU_GATE_32g.json'
    word=read(word_path);assert word['complete'] and word['independently_rechecked']
    assert all(row['speedup_percent']<0 for row in word['cases'] if row['case'] in ('decode64_4096','decode64_5120') and row['variant']!='current')
    frozen_check(gate['frozen_sources'])
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert sources==gate['frozen_sources'] and len(sources)==282 and len(profile)==54
    assert command==candidate['server_command']==control['server_command']==official_command()
    expected={key:value.replace(candidate['output_root']+'/',str(root)+'/') for key,value in candidate['profile_environment'].items()}
    assert profile==expected and profile['TIANSHU_PAIR_LOAD_MODE']=='candidate'
    assert profile['TIANSHU_FUTURE_KV_MODE']=='candidate' and 'TIANSHU_DECODE_METADATA_MODE' not in profile
    paths=[BASE/'run_validation.py',BASE/'framework_profile.py',BASE/'audit_profile.py',BASE/'batch_policy.json',BASE/'run_math_and_official.sh',
        BASE.parent/'tianshu_original_command_20261016/validation_profile.py',gate_path,word_path]
    outer={str(path):digest(path) for path in paths}
    (root/'controller_sources.json').write_text(json.dumps(outer,indent=2)+'\n')
    for name,sha in outer.items():
        target=root/'controller_source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    decision=dict(final_profile_kind='future_KV_with_paired_K_rows_warp1',runtime_source_count=282,environment_count=54,
        independent_4K_screen_gain_percent=gate['gain_percent'],word_and_warp2_not_included=True,
        fresh_Math_required=True,no_previous_Math_reused=True,screen_gate_sha256=digest(gate_path),word_gate_sha256=digest(word_path),
        runtime_sources_separate_from_controller_sources=True,only_official4K_requested=True,full_16K_not_scheduled=True)
    (root/'final_profile_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    return command,environment,profile,sources

def audit_math(root,state,profile,write=True):
    result=original_math(root,state,profile,write=False)
    result.update(final_profile_kind='future_KV_with_paired_K_rows_warp1',controller_sources=state['controller_sources'],
        runtime_source_count=282,environment_count=54,fresh_Math_for_final_runtime_profile=True,previous_accuracy_not_reused=True)
    if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def audit_benchmark(root,state,profile,quality,write=True):
    assert state['benchmark_exit_code']==0 and state['completed_runs']==4 and state['stage']=='official'
    assert quality['complete'] and quality['total']==105 and quality['accuracy']>.95 and not quality['reused_original_accuracy']
    assert quality['fresh_math_for_original_command_profile'] and quality['controller_sources']==state['controller_sources']
    assert state['server_command']==quality['server_command']==official_command()
    assert state['frozen_sources']==quality['frozen_sources']
    assert normalized_environment(state['profile_environment'],root)==normalized_environment(quality['profile_environment'],state['math_root'])
    rows=csv_rows(root,'raw_runs_*.csv');summary=csv_rows(root,'summary_*.csv')
    assert len(rows)==4 and len(summary)==1 and rows==state['raw_runs'] and summary[0]==state['official_summary']
    for row in rows:
        assert row['Run Status']=='SUCCESS' and float(row['Successful Requests'])==256
        assert [float(row[key]) for key in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
        assert float(row['Total Input Tokens'])==4096*256 and float(row['Total Output Tokens'])==1024*256
    for key in ('Total tok/s','Output tok/s','Mean TTFT (ms)'):
        assert abs(float(summary[0][key])-statistics.mean(float(row[key]) for row in rows[1:]))<.011
    assert digest(root/'source_snapshot/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')==state['official_script_sha256']==BENCH_SHA
    total=float(summary[0]['Total tok/s']);ttft=float(summary[0]['Mean TTFT (ms)']);values=[float(row['Total tok/s']) for row in rows[1:]]
    result=dict(complete=True,official_benchmark=True,original_serve_command_verified=True,adopted=False,
        final_profile_kind='future_KV_with_paired_K_rows_warp1',runtime_source_count=282,environment_count=54,
        total_tok_s=total,output_tok_s=float(summary[0]['Output tok/s']),ttft_ms=ttft,
        requests_successful=1024,requests_per_run=256,raw_runs=rows,excluded_runs=[1],
        profile=profile,controller_sources=state['controller_sources'],postfirst_spread_percent=100*(max(values)-min(values))/statistics.mean(values),
        full_16K_not_run=True,full_competition_eligibility_not_asserted=True,
        fresh_math_for_original_command_profile=True,math_correct=quality['correct'],math_total=105,math_accuracy=quality['accuracy'],
        previous_formal_4K_total_tok_s=5103.41,vs_previous_percent=100*(total/5103.41-1),
        public_baseline_total_tok_s=2028.01,vs_public_throughput_percent=100*(total/2028.01-1),
        public_ttft_limit_ms=11573.53*1.01,passes_public_4k_performance_gates=total>2028.01*1.01 and ttft<=11573.53*1.01,
        raw_response_scope='Original benchmark stdout/CSV; no full response bodies persisted by the original benchmark')
    if write:(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
