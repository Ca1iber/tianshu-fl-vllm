# SPDX-License-Identifier: Apache-2.0
import csv,importlib.util,json,math,statistics,sys
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
pair=load('_three_arm_pair_profile',EXP/'tianshu_native_int8_attention_short4k_20261022/candidate_profile.py');pa=load('_three_arm_pair_audit',EXP/'tianshu_native_int8_attention_short4k_20261022/audit_profile.py');bf=load('_three_arm_BF16_audit',EXP/'tianshu_attention_route_control_short4k_20261022/audit_profile.py')
def main():
 assert not torch.cuda.is_initialized() and not (BASE/'COMPARISON_GATE_32g.json').exists();rows=[];arms={};states={}
 roots=dict(r2='/workspace/benchmark-results/20261022_native_int8_attention_short4k_control_tianshu',candidate='/workspace/benchmark-results/20261022_native_int8_attention_short4k_candidate_tianshu',same_hook_BF16='/workspace/benchmark-results/20261022_attention_route_control_short4k_tianshu')
 for name,where in roots.items():
  root=Path(where);s=pair.read(root/'result.json');states[name]=s;assert s['phase']=='complete' and s['mode']=='candidate' and s['down_scope']=='prefill' and len(s['screen_rounds'])==3 and s['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1]);assert 'No running processes found' in (root/'hardware_after.txt').read_text()
  for field,folder in (('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')):
   pair.frozen_check(s[field])
   for p,h in s[field].items():assert pair.digest(root/folder/Path(p).relative_to('/'))==h
  actual=(bf if name=='same_hook_BF16' else pa).audit_runtime(root,s,official=True);assert actual['zero_preemptions'] and actual['actual_native_Down_runtime_banks']==42 and actual['actual_Down_M64_capture_banks']==0
  if name!='r2':
   expected=0 if name=='same_hook_BF16' else 42;assert all(v['banks']==v['M4096_banks']==expected for v in actual['actual_attention_runtime_native_by_role'].values()) and actual['actual_attention_M64_capture_native_banks']==0
  for index,r in enumerate(s['screen_rounds'],1):
   assert r['round']==index and r['successful_requests']==64 and len(r['responses'])==64 and not r['tuning_records'] and math.isfinite(r['duration_seconds']) and r['duration_seconds']>0
   assert all(v['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120) for v in r['responses'])
   total=64*5120/r['duration_seconds'];ttft=1000*statistics.mean(v['ttft_seconds'] for v in r['responses']);assert math.isclose(total,r['total_tokens_per_second'],rel_tol=1e-12) and math.isclose(ttft,1000*r['mean_ttft_seconds'],rel_tol=1e-12)
   rows.append(dict(arm=name,round=index,excluded=index==1,total_tok_s=total,output_tok_s=64*1024/r['duration_seconds'],mean_TTFT_ms=ttft))
  hot=[r for r in rows if r['arm']==name and not r['excluded']];vals=[r['total_tok_s'] for r in hot];arms[name]=dict(raw_root=where,raw_result_sha256=pair.digest(root/'result.json'),source_count=len(s['frozen_sources']),mean_total_tok_s=statistics.mean(vals),mean_TTFT_ms=statistics.mean(r['mean_TTFT_ms'] for r in hot),hot_spread_percent=100*(max(vals)-min(vals))/statistics.mean(vals),actual_runtime_audits_recomputed=True,KV_capacity_lines=[line for line in (root/'server.log').read_text(errors='replace').splitlines() if 'GPU KV cache size:' in line])
 a,b,c=states['r2'],states['candidate'],states['same_hook_BF16'];assert len(a['frozen_sources'])==890 and len(b['frozen_sources'])==903 and b['frozen_sources']==c['frozen_sources'] and all(b['frozen_sources'][p]==h for p,h in a['frozen_sources'].items());pair.quality_gate('prefill',b['frozen_sources'])
 assert a['server_command']==b['server_command']==c['server_command'] and a['additional_environment']==b['additional_environment']==c['additional_environment']
 for s in (b,c):
  ea=pair.normalized_environment(a['profile_environment'],a['output_root']);eb=pair.normalized_environment(s['profile_environment'],s['output_root']);assert {k for k in ea if ea[k]!=eb[k]}=={'PYTHONPATH'}
  for i in range(3):assert [v['input_seed'] for v in a['screen_rounds'][i]['responses']]==[v['input_seed'] for v in s['screen_rounds'][i]['responses']]
 changes={}
 for left,right in [('r2','candidate'),('r2','same_hook_BF16'),('same_hook_BF16','candidate')]:
  changes[f'{left}_to_{right}']=dict(total_throughput_gain_percent=100*(arms[right]['mean_total_tok_s']/arms[left]['mean_total_tok_s']-1),TTFT_change_percent=100*(arms[right]['mean_TTFT_ms']/arms[left]['mean_TTFT_ms']-1))
 with (BASE/'KEY_RUNS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,all9_rounds_recomputed=True,all_source_and_controller_snapshots_verified=True,all_inherited_and_new_runtime_audits_recomputed=True,same903_model_inputs_between_candidate_and_hook_control=True,fresh_quality_applies_to_candidate_only=True,control_quality_not_claimed=True,official_benchmark=False,adopted=False,requests_per_round=64,rounds=3,excluded_rounds=[1],arms=arms,changes=changes,verifier_sha256=pair.digest(__file__),plan_sha256=pair.digest(BASE/'PLAN_32g.md'),key_runs_sha256=pair.digest(BASE/'KEY_RUNS_32g.csv'));assert not torch.cuda.is_initialized();(BASE/'COMPARISON_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
