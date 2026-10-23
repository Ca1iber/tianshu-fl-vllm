# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
manifest={}
for line in (EXP/'tianshu_W4_Down64_LM1to64_release_20261023/MODEL_SOURCES.sha256').read_text().splitlines():h,p=line.split('  ',1);manifest[p]=h
assert len(manifest)==944

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def counter(p,key):return sum(float(l.split()[-1]) for l in Path(p).read_text().splitlines() if l.startswith('vllm:'+key+'{'))
records={};rows=[];canaries=[];normalized={}
for arm in ('control','candidate'):
 root=Path(f'/workspace/benchmark-results/20261023_unit_temperature_screen_v2_{arm}_tianshu');state=read(root/'result.json');assert state['phase']=='complete' and state['arm']==arm and state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1]);assert len(state['screen_rounds'])==3
 for p,h in state['frozen_sources'].items():assert sha(p)==h and sha(root/'source_snapshot'/Path(p).relative_to('/'))==h
 assert all(state['frozen_sources'][p]==h for p,h in manifest.items());assert len(state['server_command'])==13
 env=state['profile_environment']|state['additional_environment'];observed=read(root/'observed_full_server_environment.json');assert observed['environment']==env and observed['command']==state['server_command'] and len(env)==65;normalized[arm]={k:v.replace(str(root),'$RUN') for k,v in env.items()}
 route=read(next((root/'unit_records').glob('route_*.json')));assert route['mode']==arm and route['metadata_refreshes']>0 and route['unit_metadata']>0;key='skipped' if arm=='candidate' else 'unit_control';assert route['by_batch']['64'][key]>=1000 and route['policy_SHA256']==sha(EXP/'tianshu_unit_temperature_v2_20261023/policy.py')
 assert (route['skipped']==0 if arm=='control' else route['unit_control']==0)
 for r in state['screen_rounds']:
  assert len(r['responses'])==r['successful_requests']==64
  for response in r['responses']:assert response['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120) and response['text']
  assert abs(r['total_tokens_per_second']-64*5120/r['duration_seconds'])<1e-8
  if r['round']>1:assert not r['tuning_records']
  rows.append(dict(arm=arm,round=r['round'],excluded=r['round']==1,total_tok_s=r['total_tokens_per_second'],output_tok_s=r['output_tokens_per_second'],TTFT_ms=1000*r['mean_ttft_seconds'],duration_s=r['duration_seconds'],successful_requests=64,autotune_records=len(r['tuning_records'])))
 canaries.extend(r['text'] for r in state['seeded_canaries']);assert len(state['seeded_canaries'])==2
 assert 'No running processes found' in (root/'hardware_after.txt').read_text()
 preemptions=counter(root/'after_metrics.txt','num_preemptions_total')-counter(root/'before_metrics.txt','num_preemptions_total');assert preemptions==0
 hot=state['screen_rounds'][1:];values=[r['total_tokens_per_second'] for r in hot];mean=statistics.mean(values);spread=100*(max(values)-min(values))/mean;assert abs(mean-state['mean_total_tok_s'])<1e-8 and abs(spread-state['spread_percent'])<1e-8
 records[arm]=dict(raw_directory=str(root),result_sha256=sha(root/'result.json'),mean_total_tok_s=mean,mean_output_tok_s=statistics.mean(r['output_tokens_per_second'] for r in hot),TTFT_ms=1000*statistics.mean(r['mean_ttft_seconds'] for r in hot),spread_percent=spread,route=route,preemptions=preemptions,source_count=len(state['frozen_sources']),frozen_sources=state['frozen_sources'])
assert records['control']['frozen_sources']==records['candidate']['frozen_sources'];assert len(set(canaries))==1
for r in records.values():del r['frozen_sources']
different={k for k in normalized['control'] if normalized['control'][k]!=normalized['candidate'][k]};assert different=={'TIANSHU_UNIT_TEMP_MODE'},different
result=dict(complete=True,short_performance_screen_only=True,official_benchmark=False,model_quality_qualified=False,adopted=False,original_r4_model_source_count=944,all_original_and_new_sources_and_snapshots_verified=True,actual65_environment_and13_argv_verified=True,only_normalized_env_difference='TIANSHU_UNIT_TEMP_MODE',all_384_timed_requests_verified=True,cold_seeded_canaries_same_within_and_between_arms=True,all_hot_rounds_no_new_autotune=True,all_arms_no_preemption=True,old_screen_control_audit_failed_preserved=True,original_failed_observer_code_not_modified=True,full_inherited_formal_audit_not_claimed=True,arms=records,relative_total_throughput_change_percent=100*(records['candidate']['mean_total_tok_s']/records['control']['mean_total_tok_s']-1),relative_TTFT_change_percent=100*(records['candidate']['TTFT_ms']/records['control']['TTFT_ms']-1),verifier_source_sha256=sha(Path(__file__)))
(BASE/'KEY_COMPARISON_32g.json').write_text(json.dumps(result,indent=2)+'\n')
with (BASE/'KEY_ROUNDS_32g.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(result,indent=2))
