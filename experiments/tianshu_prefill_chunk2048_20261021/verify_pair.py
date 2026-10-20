# SPDX-License-Identifier: Apache-2.0
import json,statistics
from pathlib import Path
from candidate_profile import BASE,read,digest,frozen_check
from audit_profile import audit_runtime

def main():
    cpu=read(BASE/'CPU_GATE_32g.json');assert cpu['complete'];states={};roots={};rows=[];summaries={}
    for mode in ('control','candidate'):
        root=Path(f'/workspace/benchmark-results/20261021_prefill_chunk2048_{mode}_tianshu');roots[mode]=root;state=read(root/'result.json');states[mode]=state
        assert state['phase']=='complete' and state['mode']==mode and state['actual_budget']==4096
        assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
        frozen_check(state['frozen_sources']);assert len(state['frozen_sources'])==cpu['runtime_source_count']
        runtime=audit_runtime(root,state,official=True);assert runtime==state['profile_audit'] and runtime['zero_preemptions']
        for path,sha in state['frozen_sources'].items():assert digest(root/'source_snapshot'/Path(path).relative_to('/'))==sha
        for row in state['screen_rounds']:
            assert row['successful_requests']==len(row['responses'])==64
            assert abs(row['total_tokens_per_second']-64*5120/row['duration_seconds'])<1.e-8
            for response in row['responses']:assert response['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120)
            rows.append(dict(mode=mode,**{k:v for k,v in row.items() if k!='responses'}))
        assert len(state['screen_rounds'])==3
        measured=state['screen_rounds'][1:];values=[r['total_tokens_per_second'] for r in measured];mean=statistics.mean(values);spread=100*(max(values)-min(values))/mean
        assert mean==state['mean_total_tok_s'] and spread==state['spread_percent']
        tuning=[text for r in measured for text in r['tuning_records']]
        summaries[mode]=dict(total_tok_s=mean,mean_TTFT_ms=1000*statistics.mean(r['mean_ttft_seconds'] for r in measured),spread_percent=spread,
            postfirst_autotuning_records=len(tuning),postfirst_exponential_tuning_records=sum('fused_exponential_kernel' in text for text in tuning),
            warmup_elapsed_s=runtime['sampler_warmup_record']['prewarm']['elapsed_s'],actual_KV_blocks=runtime['actual_KV_blocks'],default_RNG_preserved=runtime['actual_default_RNG_preserved'],actual_prefill_threshold=runtime['actual_prefill_chunk_threshold'],max_observed_prefill_work=runtime['actual_scheduler_chunk_observation']['max_observed_prefill_work'])
    control,candidate=[states[m] for m in ('control','candidate')]
    assert control['frozen_sources']==candidate['frozen_sources'] and control['server_command']==candidate['server_command']
    expected={k:v.replace(str(roots['control'])+'/',str(roots['candidate'])+'/').replace('/hooks_control','/hooks_candidate') for k,v in control['profile_environment'].items()};assert expected==candidate['profile_environment']
    assert control['seeded_canaries'][0]['text']==control['seeded_canaries'][1]['text']==candidate['seeded_canaries'][0]['text']==candidate['seeded_canaries'][1]['text']
    gain=100*(summaries['candidate']['total_tok_s']/summaries['control']['total_tok_s']-1)
    gate=dict(complete=True,independently_rechecked=True,official_benchmark=False,adopted=False,model_Math_not_run=True,
        runtime_source_count=cpu['runtime_source_count'],environment_count=62,timed_successful_requests=384,gain_percent=gain,summaries=summaries,
        both_arms_same_model_precision_and_budget=True,seeded_canaries_identical_across_arms=True,scope='Current cache state service A/B; no assertion that all cache entries started cold',
        raw_roots={m:str(p) for m,p in roots.items()},raw_result_sha256={m:digest(p/'result.json') for m,p in roots.items()},verifier_sha256=digest(__file__))
    (BASE/'SCREEN_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'SCREEN_ROUNDS_32g.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
