# SPDX-License-Identifier: Apache-2.0
import argparse,csv,json,math,statistics,sys
from pathlib import Path
from candidate_profile import BASE,EXP,digest,read,frozen_check,quality_gate

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('prefill',),required=True);args=parser.parse_args();scope=args.scope;arms={};rows=[]
 for name in ('control','candidate'):
  root=Path(f'/workspace/benchmark-results/20261022_native_int8_attention_short4k_{name}_tianshu');state=read(root/'result.json');assert state['phase']=='complete' and state['arm']==name and state['mode']=='candidate' and state['down_scope']==scope and len(state['screen_rounds'])==3 and state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1]);frozen_check(state['frozen_sources']);frozen_check(state['controller_sources'])
  for kind,folder in [('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')]:
   for path,h in state[kind].items():assert digest(root/folder/Path(path).relative_to('/'))==h
  for index,round in enumerate(state['screen_rounds'],1):
   assert round['round']==index and round['successful_requests']==64 and len(round['responses'])==64 and not round['tuning_records'];assert all(p['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120) for p in round['responses']);duration=round['duration_seconds'];total=64*5120/duration;ttft=statistics.mean(p['ttft_seconds'] for p in round['responses']);assert math.isfinite(duration) and duration>0 and math.isclose(total,round['total_tokens_per_second'],rel_tol=1e-12);rows.append(dict(arm=name,round=index,excluded=index==1,total_tok_s=total,output_tok_s=64*1024/duration,mean_TTFT_ms=1000*ttft))
  arms[name]=dict(raw_root=str(root),raw_result_sha256=digest(root/'result.json'),mean_total_tok_s=statistics.mean(r['total_tokens_per_second'] for r in state['screen_rounds'][1:]),hot_spread_pct=100*(max(r['total_tokens_per_second'] for r in state['screen_rounds'][1:])-min(r['total_tokens_per_second'] for r in state['screen_rounds'][1:]))/statistics.mean(r['total_tokens_per_second'] for r in state['screen_rounds'][1:]),sources=state['frozen_sources'],environment=state['profile_environment'],state=state)
 a,b=arms['control'],arms['candidate'];assert len(a["sources"])==890 and len(b["sources"])==903 and all(b['sources'][p]==h for p,h in a['sources'].items());quality_gate(scope,b['sources']);assert a['state']['server_command']==b['state']['server_command'] and a['state']['additional_environment']==b['state']['additional_environment']
 for index in range(3):assert [p['input_seed'] for p in a['state']['screen_rounds'][index]['responses']]==[p['input_seed'] for p in b['state']['screen_rounds'][index]['responses']]
 from candidate_profile import normalized_environment
 ea=normalized_environment(a['environment'],a['raw_root']);eb=normalized_environment(b['environment'],b['raw_root']);assert {k for k in ea if ea[k]!=eb[k]}=={'PYTHONPATH'}
 gain=100*(b['mean_total_tok_s']/a['mean_total_tok_s']-1);passed=gain>.5 and b["hot_spread_pct"]<1 and statistics.mean(r["mean_TTFT_ms"] for r in rows if r["arm"]=="candidate" and not r["excluded"])<=1.01*statistics.mean(r["mean_TTFT_ms"] for r in rows if r["arm"]=="control" and not r["excluded"])
 with (BASE/f'KEY_RUNS_{scope}_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 gate=dict(complete=True,CPU_only=True,scope=scope,candidate_passes_perf_screen=passed,vs_exact_r2_short_gain_percent=gain,original13_argv=True,original890_sources_exact_subset=True,candidate_model_sources=903,only_environment_difference='PYTHONPATH',fresh_full_C8_C64_quality_required_and_checked=True,requests_per_round=64,rounds=3,excluded_rounds=[1],official_benchmark=False,adopted=False,arms={name:{k:v for k,v in arm.items() if k not in ('sources','environment','state')} for name,arm in arms.items()},key_runs_sha256=digest(BASE/f'KEY_RUNS_{scope}_32g.csv'));(BASE/f'SCREEN_GATE_{scope}_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate));assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
if __name__=='__main__':main()
