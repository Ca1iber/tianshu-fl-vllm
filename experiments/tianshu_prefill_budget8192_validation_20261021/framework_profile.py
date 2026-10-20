# SPDX-License-Identifier: Apache-2.0
"""Fresh Math105 and N256 official4K bound to the exact screened runtime."""
import importlib.util,json,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
SERVICE=BASE.parent/'tianshu_prefill_budget8192_service_20261021'
SCREEN=BASE.parent/'tianshu_prefill_budget8192_audit_results_20261021'
sys.path.insert(0,str(BASE.parent/'tianshu_original_command_20261016'))
from validation_profile import BENCH_SHA,audit_math as original_math,csv_rows,normalized_environment,official_command

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
parent=load('_prefill_exact_screened_profile',SERVICE/'candidate_profile.py')
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
runtime=load('_prefill_C8_or_C64_quality_audit',BASE/'audit_profile.py');audit_runtime=runtime.audit_runtime

def prepare(root,screen_root):
    assert screen_root.resolve()==SCREEN.resolve()
    pre=read(BASE/'CPU_GATE_32g.json');assert pre['complete'] and pre['runtime_source_count']==445;frozen_check(pre['sources']);frozen_check(pre['runtime_sources_unchanged'])
    gate_path=screen_root/'SCREEN_GATE_32g.json';gate=read(gate_path)
    assert gate['complete'] and gate['independently_rechecked'] and gate['environment_count']==64 and gate['runtime_source_count']==445
    assert gate['gain_percent']>=1. and gate['gain_percent']>max(r['spread_percent'] for r in gate['summaries'].values())
    assert all(r['postfirst_autotuning_records']==0 for r in gate['summaries'].values())
    candidate=read(Path(gate['raw_roots']['candidate'])/'result.json');control=read(Path(gate['raw_roots']['control'])/'result.json')
    assert gate['metadata_only_recovery'] and candidate['phase']==control['phase']=='audit_failed'
    assert candidate['error']==control['error']=="KeyError('native_sha256')" and candidate['frozen_sources']==control['frozen_sources']
    assert all(digest(Path(gate['raw_roots'][mode])/'result.json')==gate['raw_result_sha256'][mode] for mode in ('control','candidate'))
    frozen_check(candidate['frozen_sources']);command,env,profile,sources=parent.prepare(root,'candidate')
    assert sources==candidate['frozen_sources'] and len(sources)==445 and len(profile)==64
    assert command==candidate['server_command']==control['server_command']==official_command()
    expected={k:v.replace(candidate['output_root']+'/',str(root)+'/') for k,v in candidate['profile_environment'].items()};assert profile==expected
    paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'CPU_GATE_32g.json',BASE/'batch_policy.json',BASE.parent/'tianshu_original_command_20261016/validation_profile.py',gate_path,screen_root/'SCREEN_ROUNDS_32g.json',screen_root/'recovered_runtime_audit.py',screen_root/'verify_pair.py',screen_root/'CONTROL_RECOVERY_GATE_32g.json',screen_root/'CONTROLLERS_32g.sha256']
    controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
    for p,sha in controllers.items():
        target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==sha
    (root/'final_profile_decision.json').write_text(json.dumps(dict(final_profile_kind='selective_Grouped_INT8_prefill_mixed_and_NN_decode_budget8192',runtime_source_count=445,environment_count=64,screen_gain_percent=gate['gain_percent'],new_Math_required=True,previous_accuracy_not_reused=True,only_official_4K=True,full_16K_not_scheduled=True),indent=2)+'\n')
    return command,env,profile,sources

def audit_math(root,state,profile,write=True):
    result=original_math(root,state,profile,write=False)
    assert profile['actual_new_prefill_path_all_42_GateUp_verified'] and profile['verified_dense_batch']==8 and profile['LM_head_precision_in_verified_batch']=='BF16'
    result.update(final_profile_kind='selective_Grouped_INT8_prefill_mixed_and_NN_decode_budget8192',controller_sources=state['controller_sources'],runtime_source_count=445,environment_count=64,
        fresh_Math_for_final_runtime_profile=True,previous_accuracy_not_reused=True,actual_C8_GateUp_NN_integer_path_verified=True,actual_C8_LM_BF16_path_verified=True,
        actual_prefill_INT8_all_42_layers_verified=True,actual_prefill_INT8_shapes=profile['actual_prefill_INT8_shapes'],C64_quality_not_exercised_by_C8_Math=True)
    if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def audit_benchmark(root,state,profile,quality,write=True):
    assert state['benchmark_exit_code']==0 and state['completed_runs']==4 and state['stage']=='official'
    assert quality['complete'] and quality['total']==105 and quality['accuracy']>.95 and not quality['reused_original_accuracy']
    assert quality['actual_C8_GateUp_NN_integer_path_verified'] and quality['actual_C8_LM_BF16_path_verified'] and quality['actual_prefill_INT8_all_42_layers_verified'] and profile['verified_dense_batch']==64
    assert quality['controller_sources']==state['controller_sources'] and state['server_command']==quality['server_command']==official_command() and state['frozen_sources']==quality['frozen_sources']
    assert normalized_environment(state['profile_environment'],root)==normalized_environment(quality['profile_environment'],state['math_root'])
    rows=csv_rows(root,'raw_runs_*.csv');summaries=csv_rows(root,'summary_*.csv');assert len(rows)==4 and len(summaries)==1 and rows==state['raw_runs'] and summaries[0]==state['official_summary']
    for r in rows:
        assert r['Run Status']=='SUCCESS' and float(r['Successful Requests'])==256
        assert [float(r[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
        assert float(r['Total Input Tokens'])==4096*256 and float(r['Total Output Tokens'])==1024*256
    for k in ('Total tok/s','Output tok/s','Mean TTFT (ms)'):assert abs(float(summaries[0][k])-statistics.mean(float(r[k]) for r in rows[1:]))<.011
    assert digest(root/'source_snapshot/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')==state['official_script_sha256']==BENCH_SHA
    summary=summaries[0];total=float(summary['Total tok/s']);ttft=float(summary['Mean TTFT (ms)']);values=[float(r['Total tok/s']) for r in rows[1:]]
    result=dict(complete=True,official_benchmark=True,original_serve_command_verified=True,adopted=False,final_profile_kind='selective_Grouped_INT8_prefill_mixed_and_NN_decode_budget8192',runtime_source_count=445,environment_count=64,
        total_tok_s=total,output_tok_s=float(summary['Output tok/s']),ttft_ms=ttft,requests_successful=1024,requests_per_run=256,raw_runs=rows,excluded_runs=[1],profile=profile,controller_sources=state['controller_sources'],
        postfirst_spread_percent=100*(max(values)-min(values))/statistics.mean(values),fresh_math_for_original_command_profile=True,math_correct=quality['correct'],math_total=105,math_accuracy=quality['accuracy'],
        historical_recommended_4K_total_tok_s=5586.28,vs_historical_recommended_percent=100*(total/5586.28-1),historical_high_throughput_4K_total_tok_s=5509.23,vs_historical_high_throughput_percent=100*(total/5509.23-1),historical_comparison_is_not_same_run_AB=True,
        public_baseline_total_tok_s=2028.01,vs_public_throughput_percent=100*(total/2028.01-1),passes_public_4k_performance_gates=total>2028.01*1.01 and ttft<=11573.53*1.01,full_16K_not_run=True,full_competition_eligibility_not_asserted=True)
    if write:(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
