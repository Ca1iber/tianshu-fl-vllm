# SPDX-License-Identifier: Apache-2.0
import json,statistics
from pathlib import Path
from candidate_profile import BASE,read,digest,frozen_check
from audit_profile import audit_runtime

def main():
    cpu=read(BASE/'CPU_GATE_32g.json');assert cpu['complete'];states={};roots={};rows=[];summary={}
    for mode in ('control','candidate'):
        root=Path(f'/workspace/benchmark-results/20261021_prefill_key_bound_service_v3_{mode}_tianshu');roots[mode]=root;state=read(root/'result.json');states[mode]=state
        assert state['phase']=='complete' and state['mode']==mode and state['actual_budget']==4096
        assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
        frozen_check(state['frozen_sources']);assert len(state['frozen_sources'])==cpu['runtime_source_count']
        runtime=audit_runtime(root,state,official=True);assert runtime==state['profile_audit'] and runtime['zero_preemptions']
        for path,sha in state['frozen_sources'].items():assert digest(root/'source_snapshot'/Path(path).relative_to('/'))==sha
        canary=state['seeded_canaries'];assert len(canary)==2 and canary[0]['text']==canary[1]['text'] and canary[0]['cache_salt']!=canary[1]['cache_salt']
        assert all(r['api_seed']==777 and r['usage']==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144) and r['cache_salt'] for r in canary)
        assert len(state['screen_rounds'])==3
        for row in state['screen_rounds']:
            assert row['successful_requests']==len(row['responses'])==64 and abs(row['total_tokens_per_second']-64*5120/row['duration_seconds'])<1.e-8
            assert all(r['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120) and r['api_seed'] is None and r['cache_salt'] is None for r in row['responses'])
            rows.append(dict(mode=mode,**{k:v for k,v in row.items() if k!='responses'}))
        measured=state['screen_rounds'][1:];values=[r['total_tokens_per_second'] for r in measured];mean=statistics.mean(values);spread=100*(max(values)-min(values))/mean
        assert mean==state['mean_total_tok_s'] and spread==state['spread_percent']
        route=runtime['prefill_int8_record'];active=[r for r in route['routes'].values() if r['stage']=='active' and r['M']>=128 and not r['actually_inside_capture']]
        actual=[r for r in active if r['implementation']=='prefill_A8W8'];fallback=[r for r in active if r['implementation']=='delegate_qualified']
        summary[mode]=dict(total_tok_s=mean,mean_TTFT_ms=1000*statistics.mean(r['mean_ttft_seconds'] for r in measured),spread_percent=spread,postfirst_autotuning_records=sum(len(r['tuning_records']) for r in measured),actual_KV_blocks=runtime['actual_KV_blocks'],prefill_INT8_launches=sum(r['calls'] for r in actual),prefill_INT8_shapes=sorted({r['M'] for r in actual}),prefill_fallback_launches=sum(r['calls'] for r in fallback),warm_shapes=len(route['warmups']),packed_weight_payload_bytes=runtime['extra_packed_weight_bytes'])
    control,candidate=[states[m] for m in ('control','candidate')];assert control['frozen_sources']==candidate['frozen_sources'] and control['server_command']==candidate['server_command']
    assert control['seeded_canaries'][0]['text']==candidate['seeded_canaries'][0]['text']
    expected={k:v.replace(str(roots['control'])+'/',str(roots['candidate'])+'/').replace('/hooks_control','/hooks_candidate') for k,v in control['profile_environment'].items()};assert expected==candidate['profile_environment']
    gate=dict(complete=True,independently_rechecked=True,official_benchmark=False,adopted=False,new_model_Math_required=True,model_Math_not_run=True,runtime_source_count=cpu['runtime_source_count'],environment_count=62,timed_successful_requests=384,
        gain_percent=100*(summary['candidate']['total_tok_s']/summary['control']['total_tok_s']-1),TTFT_change_percent=100*(summary['candidate']['mean_TTFT_ms']/summary['control']['mean_TTFT_ms']-1),summaries=summary,
        same_NN_decode_and_packed_weight_layout_in_both_arms=True,only_prefill_stage_key_bound_changed=True,actual_effect_exercised=bool(candidate['profile_audit']['actual_tightened_rows_observed']),untimed_canaries_unique_cache_salt_for_full_recompute=True,seed_reproducible_within_each_arm=True,cross_arm_text_equality_is_additional_canary_not_Math=True,timed_requests_no_seed_or_cache_salt=True,
        raw_roots={m:str(p) for m,p in roots.items()},raw_result_sha256={m:digest(p/'result.json') for m,p in roots.items()},verifier_sha256=digest(__file__))
    (BASE/'SCREEN_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'SCREEN_ROUNDS_32g.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
