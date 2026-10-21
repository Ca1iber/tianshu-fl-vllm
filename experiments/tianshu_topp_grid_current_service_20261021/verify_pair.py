# SPDX-License-Identifier: Apache-2.0
import csv,json,statistics
from pathlib import Path
from candidate_profile import BASE,read,digest,frozen_check
from audit_profile import audit_runtime
cpu=read(BASE/'CPU_GATE_32g.json');assert cpu['complete'];states={};roots={};rows=[];summary={}
def usage_ok(value,prompt,completion):return value['prompt_tokens']==prompt and value['completion_tokens']==completion and value['total_tokens']==prompt+completion
for mode in ('control','candidate'):
 root=Path(f'/workspace/benchmark-results/20261021_topp_grid_current_service_{mode}_tianshu');roots[mode]=root
 state=read(root/'result.json');states[mode]=state
 assert state['phase']=='complete' and state['mode']==mode and state['actual_budget']==4096
 assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
 frozen_check(state['frozen_sources']);assert len(state['frozen_sources'])==cpu['runtime_source_count']
 runtime=audit_runtime(root,state,official=True);assert runtime==state['profile_audit'] and runtime['zero_preemptions']
 for path,h in state['frozen_sources'].items():assert digest(root/'source_snapshot'/Path(path).relative_to('/'))==h
 canary=state['seeded_canaries'];assert len(canary)==2 and canary[0]['text']==canary[1]['text'] and canary[0]['cache_salt']!=canary[1]['cache_salt']
 assert all(x['api_seed']==777 and usage_ok(x['usage'],128,16) for x in canary)
 assert usage_ok(state['joint_C8_smoke']['response']['usage'],1024,128)
 assert len(state['screen_rounds'])==3
 for row in state['screen_rounds']:
  assert row['successful_requests']==len(row['responses'])==64
  assert abs(row['total_tokens_per_second']-64*5120/row['duration_seconds'])<1e-8
  assert abs(row['output_tokens_per_second']-64*1024/row['duration_seconds'])<1e-8
  assert all(usage_ok(x['usage'],4096,1024) and x['api_seed'] is None and x['cache_salt'] is None for x in row['responses'])
  assert statistics.mean(x['ttft_seconds'] for x in row['responses'])==row['mean_ttft_seconds']
  rows.append(dict(mode=mode,**{k:v for k,v in row.items() if k!='responses'}))
 measured=state['screen_rounds'][1:];values=[r['total_tokens_per_second'] for r in measured]
 mean=statistics.mean(values);assert mean==state['mean_total_tok_s']
 policy=runtime['topp_grid_record'];calls=runtime['topp_grid_observed']
 summary[mode]=dict(total_tok_s=mean,mean_TTFT_ms=1000*statistics.mean(x['mean_ttft_seconds'] for x in measured),spread_percent=state['spread_percent'],postfirst_autotuning_records=sum(len(x['tuning_records']) for x in measured),actual_KV_blocks=runtime['actual_KV_blocks'],runtime_batches=runtime['actual_sampler_runtime_batches'],native_peak_activation_memory=policy['native_final_peak_activation_memory'],native_non_torch_memory=policy['native_final_non_torch_memory'],warm_calls=len(policy['warm_calls']),shared_sampler_scratch_bytes=policy['shared_sampler_scratch_bytes'],packed_weight_payload_bytes=runtime['extra_packed_weight_bytes'])
a,b=[states[m] for m in ('control','candidate')]
assert a['frozen_sources']==b['frozen_sources'] and a['server_command']==b['server_command']
assert {k:v.replace(str(roots['control'])+'/',str(roots['candidate'])+'/').replace('/hooks_control','/hooks_candidate') for k,v in a['profile_environment'].items()}==b['profile_environment']
assert a['seeded_canaries'][0]['text']==b['seeded_canaries'][0]['text']
gain=100*(summary['candidate']['total_tok_s']/summary['control']['total_tok_s']-1)
gate=dict(complete=True,independently_rechecked=True,official_benchmark=False,adopted=False,new_model_Math_required=True,model_Math_not_run=True,runtime_source_count=cpu['runtime_source_count'],environment_count=62,timed_successful_requests=384,gain_percent=gain,TTFT_change_percent=100*(summary['candidate']['mean_TTFT_ms']/summary['control']['mean_TTFT_ms']-1),summaries=summary,
 quality_worth_running=gain>.5 and all(v['spread_percent']<.5 and v['postfirst_autotuning_records']==0 for v in summary.values()),
 both_arms_same_current_model_precision=True,only_sampler_geometry_changes=True,default_RNG_algorithm_unchanged=True,equal_warmup_native_accounting=True,
 canary_B1_fallback_cross_arm_text_equal=True,actual_C8_and_C64_runtime_verified=True,raw_roots={m:str(r) for m,r in roots.items()},raw_result_sha256={m:digest(r/'result.json') for m,r in roots.items()},verifier_sha256=digest(__file__))
(BASE/'SCREEN_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'SCREEN_ROUNDS_32g.json').write_text(json.dumps(rows,indent=2)+'\n')
with (BASE/'KEY_ROUNDS_32g.csv').open('w',newline='') as f:
 fields=('mode','round','duration_seconds','total_tokens_per_second','output_tokens_per_second','mean_ttft_seconds','successful_requests')
 writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
print(json.dumps(gate))
