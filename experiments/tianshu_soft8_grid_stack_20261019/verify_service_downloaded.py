# SPDX-License-Identifier: Apache-2.0
"""Audit smoke or paired 4K response/timing/source evidence without inventing quality."""
import argparse,hashlib,json,re,statistics
from pathlib import Path
import importlib.util
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_soft8_offline_audit',BASE/'audit_profile.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();root=a.root.resolve()
def read(p):return json.loads(p.read_text())
repo=BASE.parents[1]
comparison=None if a.smoke else read(root/'comparison.json')
if comparison is not None:assert comparison['complete']
modes={};response_count=0;metric_count=0;timings={};tunes={}
for mode in (('candidate',) if a.smoke else ('control','candidate')):
 run=root if a.smoke else root/mode;s=read(run/'result.json');assert s['phase']=='complete' and s['mode']==mode
 profile=audit.audit_runtime(run,s,official=not a.smoke,online=False);assert profile['software8_KV'] and profile['TopP_16K_grid2_API']==(mode=='candidate')
 for source,h in s['frozen_sources'].items():
  if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):assert hashlib.sha256((repo/Path(source).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==h
 w=s['workload'];n,il,ol,nrounds=w['requests_per_round'],w['input_tokens'],w['output_tokens'],w['rounds']
 assert (n,il,ol,nrounds)==((8,512,128,1) if a.smoke else (64,4096,1024,3))
 assert w['excluded_rounds']==([] if a.smoke else [1])
 rows=s['short_4k_rounds'];assert len(rows)==nrounds
 seeds=[]
 for i,row in enumerate(rows):
  rs=row['responses'];assert row['round']==i+1 and row['successful_requests']==len(rs)==n
  expected=[202610193000+100*i+j for j in range(n)];assert [r['input_seed'] for r in rs]==expected;seeds.extend(expected)
  for r in rs:
   assert r['api_seed'] is None and (r['input_tokens'],r['output_tokens'])==(il,ol)
   assert r['usage']['prompt_tokens']==il and r['usage']['completion_tokens']==ol and r['usage']['total_tokens']==il+ol
   assert isinstance(r['text'],str) and r['text'] and 0<r['ttft_seconds']<=r['wall_seconds']<=row['duration_seconds']+.1
  assert abs(row['total_tokens_per_second']-n*(il+ol)/row['duration_seconds'])<1e-9
  assert abs(row['output_tokens_per_second']-n*ol/row['duration_seconds'])<1e-9
  assert abs(row['mean_ttft_seconds']-statistics.mean(r['ttft_seconds'] for r in rs))<1e-12
  response_count+=len(rs)
 assert len(set(seeds))==n*nrounds
 steady=rows if a.smoke else rows[1:];mean=statistics.mean(r['total_tokens_per_second'] for r in steady)
 assert abs(s['mean_total_tok_s']-mean)<1e-9
 spread=100*(max(r['total_tokens_per_second'] for r in steady)-min(r['total_tokens_per_second'] for r in steady))/mean
 assert abs(s['spread_percent']-spread)<1e-10
 can=s['seeded_canaries'];assert len(can)==2 and can[0]['text']==can[1]['text']
 for c in can:assert c['api_seed']==777 and c['input_seed']==2026101699 and c['usage']['prompt_tokens']==128 and c['usage']['completion_tokens']==16
 metrics=list(run.glob('*metrics.txt'));assert len(metrics)==2+2*nrounds
 for m in metrics:
  v=re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$',m.read_text(),re.M);assert v and all(float(x)==0 for x in v);metric_count+=1
 modes[mode]=s;timings[mode]=[r['total_tokens_per_second'] for r in rows];tunes[mode]=[len(r['tuning_records']) for r in rows]
if not a.smoke:
 c,d=modes['control'],modes['candidate'];assert c['frozen_sources']==d['frozen_sources']==comparison['frozen_sources'] and c['server_command']==d['server_command']
 for key,value in c['profile_environment'].items():
  other=d['profile_environment'][key]
  if key=='TIANSHU_TOPP_GRID_CANDIDATE':assert (value,other)==('0','1')
  else:assert value.replace(c['output_root'],'<RUN>')==other.replace(d['output_root'],'<RUN>'),key
 assert comparison['control_total_tok_s']==c['mean_total_tok_s'] and comparison['candidate_total_tok_s']==d['mean_total_tok_s']
 assert comparison['gain_percent']==100*(d['mean_total_tok_s']/c['mean_total_tok_s']-1)
r=dict(complete=True,smoke=a.smoke,response_texts_usage_seeds_and_timings_checked=response_count,within_mode_seeded_canaries=2*len(modes),metrics_zero_preemptions=metric_count,source_snapshots=len(modes['candidate']['frozen_sources']),actual_UInt8_spec_and_cache_view_layers=42,bounded_scratch_accounted=True,sampler_algorithm_unchanged=True,KV_values_quantized=True,timings_total_tok_s=timings,tuning_records_per_round=tunes,official_benchmark=False,fresh_Math_not_run=True,adopted=False)
if comparison is not None:r.update(control_total_tok_s=comparison['control_total_tok_s'],candidate_total_tok_s=comparison['candidate_total_tok_s'],gain_percent=comparison['gain_percent'])
(root/'local_download_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
