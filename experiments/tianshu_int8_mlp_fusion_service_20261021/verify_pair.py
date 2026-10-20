# SPDX-License-Identifier: Apache-2.0
"""Independent source, executed-path, banks and all six screen rounds audit."""
import hashlib,json,statistics
from pathlib import Path
from candidate_profile import BASE,read,digest,frozen_check
from audit_profile import audit_runtime

def signature(runtime):
    return sorted([{k:r[k] for k in ('role','prefix','shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes')} for r in runtime['dense_int8_record']['banks']],key=lambda r:r['prefix'])
def write(name,value):(BASE/name).write_text(json.dumps(value,indent=2)+'\n')
def main():
    cpu=read(BASE/'CPU_GATE_32g.json');assert cpu['complete'] and cpu['no_GPU_started']
    states={};runtimes={};rows=[];summaries={}
    roots={mode:Path(f'/workspace/benchmark-results/20261021_int8_mlp_fusion_service_{mode}_tianshu') for mode in ('control','candidate')}
    expected_banks=read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')
    for mode,root in roots.items():
        state=read(root/'result.json');states[mode]=state
        assert state['phase']=='complete' and state['mode']==mode and state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
        assert len(state['profile_environment'])==63 and len(state['frozen_sources'])==cpu['runtime_source_count']
        frozen_check(state['frozen_sources'])
        for path,sha in state['frozen_sources'].items():assert digest(root/'source_snapshot'/Path(path).relative_to('/'))==sha
        runtime=audit_runtime(root,state,official=True);assert runtime==state['profile_audit'];runtimes[mode]=runtime
        assert signature(runtime)==expected_banks and runtime['all_43_banks_same_as_qualified_Math'] and runtime['zero_preemptions']
        assert len(state['seeded_canaries'])==2 and state['seeded_canaries'][0]['text']==state['seeded_canaries'][1]['text'] and len(state['int8_smoke_wave'])==8
        assert len(state['screen_rounds'])==3
        for round in state['screen_rounds']:
            assert round['successful_requests']==len(round['responses'])==64 and round['round'] in (1,2,3)
            assert abs(round['total_tokens_per_second']-64*5120/round['duration_seconds'])<1.e-8
            assert abs(round['output_tokens_per_second']-64*1024/round['duration_seconds'])<1.e-8
            for response in round['responses']:
                assert response['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120)
                assert response['input_tokens']==4096 and response['output_tokens']==1024 and response['ttft_seconds']>0
            assert abs(round['mean_ttft_seconds']-statistics.mean(r['ttft_seconds'] for r in round['responses']))<1.e-9
            rows.append(dict(mode=mode,**{k:v for k,v in round.items() if k!='responses'}))
        counted=state['screen_rounds'][1:];values=[r['total_tokens_per_second'] for r in counted]
        mean=statistics.mean(values);spread=100*(max(values)-min(values))/mean
        assert mean==state['mean_total_tok_s'] and spread==state['spread_percent']
        summaries[mode]=dict(mean_total_tok_s=mean,mean_TTFT_ms=1000*statistics.mean(r['mean_ttft_seconds'] for r in counted),spread_percent=spread,
            postfirst_autotuning_records=sum(len(r['tuning_records']) for r in counted),actual_KV_blocks=runtime['actual_KV_blocks'],
            capture_memory=runtime['capture_memory'],actual_fused_prefix_counts=runtime['actual_fused_capture_and_model_replay_prefix_counts'],
            actual_binary_matches_component=runtime['actual_binary_matches_component'],
            selected_activation_sources=runtime['selected_activation_sources'])
    control,candidate=[states[m] for m in ('control','candidate')]
    assert control['frozen_sources']==candidate['frozen_sources']
    assert control['server_command']==candidate['server_command'] and len(control['server_command'])==13
    expected={key:value.replace(str(roots['control'])+'/',str(roots['candidate'])+'/') for key,value in control['profile_environment'].items()}
    expected['TIANSHU_MLP_FUSION_MODE']='candidate';assert expected==candidate['profile_environment']
    reference_root=Path("/workspace/benchmark-results/20261020_lm_precision_service_candidate_tianshu")
    reference=read(reference_root/"result.json");assert reference["phase"]=="complete" and reference["mean_total_tok_s"]>5400
    for path,sha in reference["frozen_sources"].items():assert candidate["frozen_sources"][path]==sha
    reference_gain=100*(summaries["candidate"]["mean_total_tok_s"]/reference["mean_total_tok_s"]-1)
    gain=100*(summaries['candidate']['mean_total_tok_s']/summaries['control']['mean_total_tok_s']-1)
    ttft_change=100*(summaries['candidate']['mean_TTFT_ms']/summaries['control']['mean_TTFT_ms']-1)
    worth=bool(gain>=.5 and reference_gain>=.5 and ttft_change<=1 and all(r['spread_percent']<1 and r['postfirst_autotuning_records']==0 for r in summaries.values()) and all(v==42 for v in runtimes['candidate']['actual_fused_capture_and_model_replay_prefix_counts'].values()))
    gate=dict(complete=True,independently_rechecked=True,official_benchmark=False,adopted=False,quality_worth_running=worth,
        raw_roots={k:str(v) for k,v in roots.items()},runtime_source_count=cpu['runtime_source_count'],environment_count=63,
        requests_per_round=64,rounds_per_arm=3,excluded_rounds=[1],timed_successful_requests=384,gain_percent=gain,TTFT_change_percent=ttft_change,
        postfirst_no_runtime_autotuning=all(r['postfirst_autotuning_records']==0 for r in summaries.values()),summaries=summaries,
        prior_same_client_small_reference_root=str(reference_root),prior_small_reference_total_tok_s=reference["mean_total_tok_s"],vs_prior_small_reference_percent=reference_gain,
        MLP_wrapper_changes_compile_boundary=True,new_Math_required=True,old_Math_not_reused=True,
        all_43_banks_match_qualified_Math=True,LM_small_BF16_and_LM64_integer_retained=True,
        actual_GateUp_C8_and_C64_capture_and_replay_verified=True,actual_fused_42_layers_C8_C64=all(v==42 for v in runtimes['candidate']['actual_fused_capture_and_model_replay_prefix_counts'].values()),
        full_16K_not_run=True,fresh_Math_not_run=True,previous_qualified_original_4K_total_tok_s=5482.79,
        frozen_sources=candidate['frozen_sources'],raw_result_sha256={mode:digest(root/'result.json') for mode,root in roots.items()},verifier_sha256=digest(__file__))
    write('SCREEN_GATE_32g.json',gate);write('SCREEN_ROUNDS_32g.json',rows)
    print(json.dumps({k:v for k,v in gate.items() if k!='frozen_sources'}))
if __name__=='__main__':main()
