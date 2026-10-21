# SPDX-License-Identifier: Apache-2.0
import json,statistics
from pathlib import Path
from candidate_profile import BASE,read,digest,frozen_check
from audit_profile import audit_runtime
cpu=read(BASE/'CPU_GATE_32g.json');assert cpu['complete'];states={};roots={};rows=[];summary={}
for mode in ('control','candidate'):
 root=Path(f'/workspace/benchmark-results/20261021_topp_warp_service_{mode}_tianshu');roots[mode]=root;state=read(root/'result.json');states[mode]=state
 assert state['phase']=='complete' and state['mode']==mode and state['actual_budget']==4096
 assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
 frozen_check(state['frozen_sources']);assert len(state['frozen_sources'])==cpu['runtime_source_count']
 runtime=audit_runtime(root,state,official=True);assert runtime==state['profile_audit'] and runtime['zero_preemptions']
 for path,h in state['frozen_sources'].items():assert digest(root/'source_snapshot'/Path(path).relative_to('/'))==h
 canary=state['seeded_canaries'];assert len(canary)==2 and canary[0]['text']==canary[1]['text'] and canary[0]['cache_salt']!=canary[1]['cache_salt']
 assert all(x['api_seed']==777 and x['usage']==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144) for x in canary)
 assert len(state['screen_rounds'])==3
 for row in state['screen_rounds']:
  assert row['successful_requests']==len(row['responses'])==64 and abs(row['total_tokens_per_second']-64*5120/row['duration_seconds'])<1e-8
  assert all(x['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120) and x['api_seed'] is None and x['cache_salt'] is None for x in row['responses'])
  rows.append(dict(mode=mode,**{k:v for k,v in row.items() if k!='responses'}))
 measured=state['screen_rounds'][1:];values=[r['total_tokens_per_second'] for r in measured];mean=statistics.mean(values);assert mean==state['mean_total_tok_s']
 route=runtime['topp_warp_record'];active=[r for r in route['routes'].values() if r['stage']=='runtime']
 summary[mode]=dict(total_tok_s=mean,mean_TTFT_ms=1000*statistics.mean(x['mean_ttft_seconds'] for x in measured),spread_percent=state['spread_percent'],postfirst_autotuning_records=sum(len(x['tuning_records']) for x in measured),actual_KV_blocks=runtime['actual_KV_blocks'],sampler_warps=runtime['actual_topp_runtime_warps'],runtime_sampler_calls_recorded=sum(x['calls'] for x in active),runtime_batches=sorted({x['batch'] for x in active}),native_peak_activation_memory=route['native_final_peak_activation_memory'],native_non_torch_memory=route['native_final_non_torch_memory'],warm_calls=len(route['warm_calls']),sampler_buffer_bytes=route['native_sampler_buffer_bytes'],packed_weight_payload_bytes=runtime['extra_packed_weight_bytes'])
a,b=[states[m] for m in ('control','candidate')]
assert a['frozen_sources']==b['frozen_sources'] and a['server_command']==b['server_command']
assert {k:v.replace(str(roots['control'])+'/',str(roots['candidate'])+'/').replace('/hooks_control','/hooks_candidate') for k,v in a['profile_environment'].items()}==b['profile_environment']
assert a['seeded_canaries'][0]['text']==b['seeded_canaries'][0]['text']
gate=dict(complete=True,independently_rechecked=True,official_benchmark=False,adopted=False,new_model_Math_required=True,model_Math_not_run=True,runtime_source_count=cpu['runtime_source_count'],environment_count=62,timed_successful_requests=384,gain_percent=100*(summary['candidate']['total_tok_s']/summary['control']['total_tok_s']-1),TTFT_change_percent=100*(summary['candidate']['mean_TTFT_ms']/summary['control']['mean_TTFT_ms']-1),summaries=summary,both_arms_same_current_model_precision=True,only_topp_warp_configuration_changes_at_runtime=True,FP32_reduction_differences_explicit=True,default_RNG_algorithm_unchanged=True,default_RNG_unchanged_by_equal_warmups=True,canary_B1_fallback_cross_arm_text_equal=True,raw_roots={m:str(r) for m,r in roots.items()},raw_result_sha256={m:digest(r/'result.json') for m,r in roots.items()},verifier_sha256=digest(__file__))
(BASE/'SCREEN_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'SCREEN_ROUNDS_32g.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(gate))
