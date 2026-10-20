# SPDX-License-Identifier: Apache-2.0
"""Fresh Math and original N256 4K for the exact screened prefill-linear profile."""
import importlib.util,json,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_original_command_20261016'))
from validation_profile import BENCH_SHA,audit_math as original_math,csv_rows,normalized_environment,official_command

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
parent=load('_prefill_final_service_profile',BASE.parent/'tianshu_dense_budget8192_v2_20261021/candidate_profile.py')
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
runtime=load('_prefill_final_runtime',BASE/'audit_profile.py');audit_runtime=runtime.audit_runtime

def prepare(root,screen_root):
    gate_path=screen_root/'SCREEN_GATE_32g.json';gate=read(gate_path)
    assert gate['complete'] and gate['independently_rechecked'] and gate['quality_worth_running']
    assert gate['all_43_weight_banks_match_qualified_profile'] and gate['environment_count']==64 and gate['runtime_source_count']==374
    assert gate['requests_per_round']==64 and gate['excluded_rounds']==[1] and gate['postfirst_no_runtime_autotuning']
    candidate=read(Path(gate['raw_roots']['candidate'])/'result.json');control=read(Path(gate['raw_roots']['control'])/'result.json')
    assert candidate['phase']==control['phase']=='complete'
    assert candidate['profile_audit']['actual_ready_request_path_all_42_GateUp_verified'] and candidate['profile_audit']['actual_worker_and_scheduler_budget_verified']
    assert candidate['actual_budget']==8192 and control['actual_budget']==4096
    assert candidate['profile_audit']['warm_points']==99 and control['profile_audit']['warm_points']==83
    assert candidate['frozen_sources']==control['frozen_sources']==gate['frozen_sources']
    frozen_check(gate['frozen_sources'])
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert sources==gate['frozen_sources'] and len(sources)==374 and len(profile)==64
    assert command==candidate['server_command']==control['server_command']==official_command()
    expected={key:value.replace(candidate['output_root']+'/',str(root)+'/') for key,value in candidate['profile_environment'].items()}
    assert profile==expected and profile['TIANSHU_PREFILL_LINEAR_MODE']=='candidate' and profile['TIANSHU_COLD_START_MODE']=='gpu4'
    assert profile["TIANSHU_LM_PRECISION_MODE"]=="candidate"
    assert profile['TIANSHU_DENSE_BUDGET_MODE']=='candidate'
    paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'batch_policy.json',BASE.parent/'tianshu_original_command_20261016/validation_profile.py',gate_path]
    outer={str(path):digest(path) for path in paths};(root/'controller_sources.json').write_text(json.dumps(outer,indent=2)+'\n')
    for name,sha in outer.items():
        target=root/'controller_source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    (root/'final_profile_decision.json').write_text(json.dumps(dict(final_profile_kind='W8A8_GateUp_LM64_BF16_LM_small_and_prefill_linear_budget8192',runtime_source_count=374,environment_count=64,
        independent_screen_gain_percent=gate['gain_percent'],all_42_new_prefill_paths_actually_executed=True,fresh_Math_required=True,
        no_previous_Math_reused=True,only_official4K_requested=True,full_16K_not_scheduled=True),indent=2)+'\n')
    return command,environment,profile,sources

def audit_math(root,state,profile,write=True):
    result=original_math(root,state,profile,write=False)
    assert profile['actual_new_prefill_path_all_42_GateUp_verified'] and profile['verified_dense_batch']==8
    result.update(final_profile_kind='W8A8_GateUp_LM64_BF16_LM_small_and_prefill_linear_budget8192',controller_sources=state['controller_sources'],runtime_source_count=374,environment_count=64,
        fresh_Math_for_final_runtime_profile=True,previous_accuracy_not_reused=True,actual_C8_GateUp_integer_path_verified=True,actual_C8_LM_BF16_path_verified=profile["LM_head_precision_in_verified_batch"]=="BF16",LM_M64_integer_not_exercised_by_C8_Math=True,actual_new_prefill_path_all_42_GateUp_verified=True)
    if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def audit_benchmark(root,state,profile,quality,write=True):
    assert state['benchmark_exit_code']==0 and state['completed_runs']==4 and state['stage']=='official'
    assert quality['complete'] and quality['total']==105 and quality['accuracy']>.95 and not quality['reused_original_accuracy']
    assert quality["actual_C8_GateUp_integer_path_verified"] and quality["actual_C8_LM_BF16_path_verified"] and quality['actual_new_prefill_path_all_42_GateUp_verified'] and profile['verified_dense_batch']==64
    assert quality['controller_sources']==state['controller_sources'] and state['server_command']==quality['server_command']==official_command()
    assert state['frozen_sources']==quality['frozen_sources']
    assert normalized_environment(state['profile_environment'],root)==normalized_environment(quality['profile_environment'],state['math_root'])
    rows=csv_rows(root,'raw_runs_*.csv');summary=csv_rows(root,'summary_*.csv')
    assert len(rows)==4 and len(summary)==1 and rows==state['raw_runs'] and summary[0]==state['official_summary']
    for row in rows:
        assert row['Run Status']=='SUCCESS' and float(row['Successful Requests'])==256
        assert [float(row[key]) for key in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
        assert float(row['Total Input Tokens'])==4096*256 and float(row['Total Output Tokens'])==1024*256
    for key in ('Total tok/s','Output tok/s','Mean TTFT (ms)'):assert abs(float(summary[0][key])-statistics.mean(float(row[key]) for row in rows[1:]))<.011
    assert digest(root/'source_snapshot/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')==state['official_script_sha256']==BENCH_SHA
    total=float(summary[0]['Total tok/s']);ttft=float(summary[0]['Mean TTFT (ms)']);values=[float(row['Total tok/s']) for row in rows[1:]]
    result=dict(complete=True,official_benchmark=True,original_serve_command_verified=True,adopted=False,final_profile_kind='W8A8_GateUp_LM64_BF16_LM_small_and_prefill_linear_budget8192',
        runtime_source_count=374,environment_count=64,total_tok_s=total,output_tok_s=float(summary[0]['Output tok/s']),ttft_ms=ttft,
        requests_successful=1024,requests_per_run=256,raw_runs=rows,excluded_runs=[1],profile=profile,controller_sources=state['controller_sources'],
        postfirst_spread_percent=100*(max(values)-min(values))/statistics.mean(values),fresh_math_for_original_command_profile=True,
        math_correct=quality['correct'],math_total=105,math_accuracy=quality['accuracy'],previous_formal_4K_total_tok_s=5482.79,vs_previous_percent=100*(total/5482.79-1),
        previous_formal_4K_ttft_ms=4015.23,passes_previous_1percent_gate=total>5482.79*1.01 and ttft<=4015.23*1.01,
        public_baseline_total_tok_s=2028.01,vs_public_throughput_percent=100*(total/2028.01-1),public_ttft_limit_ms=11573.53*1.01,
        passes_public_4k_performance_gates=total>2028.01*1.01 and ttft<=11573.53*1.01,full_16K_not_run=True,full_competition_eligibility_not_asserted=True,
        raw_response_scope='Original benchmark stdout/CSV; no full response bodies persisted by original benchmark')
    if write:(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
