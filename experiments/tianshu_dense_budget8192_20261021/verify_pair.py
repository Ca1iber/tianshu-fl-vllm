# SPDX-License-Identifier: Apache-2.0
import json,statistics
from pathlib import Path
from candidate_profile import BASE,read,digest,frozen_check
from audit_profile import audit_runtime

def signature(runtime):
    return sorted([{k:r[k] for k in ('role','prefix','shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes')} for r in runtime['dense_int8_record']['banks']],key=lambda r:r['prefix'])
def write(name,value):(BASE/name).write_text(json.dumps(value,indent=2)+'\n')
def main():
    cpu=read(BASE/'CPU_GATE_32g.json');assert cpu['complete'] and cpu['no_GPU_started']
    roots={mode:Path(f'/workspace/benchmark-results/20261021_dense_budget8192_service_{mode}_tianshu') for mode in ('control','candidate')}
    states={};summaries={};rounds=[]
    expected_banks=read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')
    for mode,root in roots.items():
        state=read(root/'result.json');states[mode]=state
        assert state['phase']=='complete' and state['mode']==mode and state['actual_budget']==(4096 if mode=='control' else 8192)
        assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
        assert len(state['frozen_sources'])==cpu['runtime_source_count'] and len(state['profile_environment'])==64
        frozen_check(state['frozen_sources'])
        for path,sha in state['frozen_sources'].items():assert digest(root/'source_snapshot'/Path(path).relative_to('/'))==sha
        runtime=audit_runtime(root,state,official=True);assert runtime==state['profile_audit']
        assert signature(runtime)==expected_banks and runtime['zero_preemptions'] and runtime['actual_worker_and_scheduler_budget_verified']
        assert len(state['seeded_canaries'])==2 and state['seeded_canaries'][0]['text']==state['seeded_canaries'][1]['text'] and len(state['int8_smoke_wave'])==8
        assert len(state['screen_rounds'])==3
        for row in state['screen_rounds']:
            assert len(row['responses'])==row['successful_requests']==64
            assert abs(row['total_tokens_per_second']-64*5120/row['duration_seconds'])<1.e-8
            assert abs(row['output_tokens_per_second']-64*1024/row['duration_seconds'])<1.e-8
            for response in row['responses']:
                assert response['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120)
                assert response['input_tokens']==4096 and response['output_tokens']==1024 and response['ttft_seconds']>0
            assert abs(row['mean_ttft_seconds']-statistics.mean(r['ttft_seconds'] for r in row['responses']))<1.e-9
            rounds.append(dict(mode=mode,**{k:v for k,v in row.items() if k!='responses'}))
        counted=state['screen_rounds'][1:];values=[row['total_tokens_per_second'] for row in counted]
        mean=statistics.mean(values);spread=100*(max(values)-min(values))/mean
        assert mean==state['mean_total_tok_s'] and spread==state['spread_percent']
        summaries[mode]=dict(mean_total_tok_s=mean,spread_percent=spread,mean_TTFT_ms=1000*statistics.mean(r['mean_ttft_seconds'] for r in counted),postfirst_autotuning_records=sum(len(r['tuning_records']) for r in counted),actual_KV_blocks=runtime['actual_KV_blocks'],actual_budget=runtime['scheduler_budget'],warm_points=runtime['warm_points'])
    control,candidate=[states[m] for m in ('control','candidate')]
    assert control['frozen_sources']==candidate['frozen_sources'] and control['server_command']==candidate['server_command'] and len(control['server_command'])==13
    expected={key:value.replace(str(roots['control'])+'/',str(roots['candidate'])+'/') for key,value in control['profile_environment'].items()}
    expected['TIANSHU_DENSE_BUDGET_MODE']='candidate';assert expected==candidate['profile_environment']
    reference_root=Path('/workspace/benchmark-results/20261020_lm_precision_service_candidate_tianshu');reference=read(reference_root/'result.json');assert reference['phase']=='complete'
    for path,sha in reference['frozen_sources'].items():assert candidate['frozen_sources'][path]==sha
    gain=100*(summaries['candidate']['mean_total_tok_s']/summaries['control']['mean_total_tok_s']-1)
    prior_gain=100*(summaries['candidate']['mean_total_tok_s']/reference['mean_total_tok_s']-1)
    ttft=100*(summaries['candidate']['mean_TTFT_ms']/summaries['control']['mean_TTFT_ms']-1)
    worth=bool(gain>=.5 and prior_gain>=.5 and ttft<=1 and all(r['spread_percent']<1 and r['postfirst_autotuning_records']==0 for r in summaries.values()))
    gate=dict(complete=True,independently_rechecked=True,official_benchmark=False,adopted=False,quality_worth_running=worth,
        requests_per_round=64,rounds_per_arm=3,excluded_rounds=[1],timed_successful_requests=384,
        runtime_source_count=cpu['runtime_source_count'],environment_count=64,gain_percent=gain,TTFT_change_percent=ttft,
        prior_same_client_small_reference_total_tok_s=reference['mean_total_tok_s'],vs_prior_small_reference_percent=prior_gain,
        summaries=summaries,raw_roots={m:str(r) for m,r in roots.items()},frozen_sources=candidate['frozen_sources'],
        all_43_weight_banks_match_qualified_profile=True,LM_precision_and_GateUp_policy_unchanged=True,
        postfirst_no_runtime_autotuning=all(r['postfirst_autotuning_records']==0 for r in summaries.values()),fresh_Math_not_run=True,new_Math_required=True,full_16K_not_run=True,
        raw_result_sha256={m:digest(r/'result.json') for m,r in roots.items()},verifier_sha256=digest(__file__))
    write('SCREEN_GATE_32g.json',gate);write('SCREEN_ROUNDS_32g.json',rounds);print(json.dumps({k:v for k,v in gate.items() if k!='frozen_sources'}))
if __name__=='__main__':main()
