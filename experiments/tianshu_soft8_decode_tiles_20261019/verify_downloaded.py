# SPDX-License-Identifier: Apache-2.0
"""Check frozen-source and recorded numerical/timing evidence, not model quality."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from plans import PLANS
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
d=json.loads((root/'result.json').read_text());assert d['complete'] and d['phase']=='complete' and d['writer_included']
assert not d['official_benchmark'] and not d['adopted'] and d['plans']==PLANS and len(d['profile_environment'])==38
repo=Path(__file__).resolve().parents[2]
for name,h in d['source_snapshots'].items():
 assert hashlib.sha256((root/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==h
 if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==h
assert len(d['source_snapshots'])==109 and len(d['cases'])==10 and len(d['events'])==630
labels=['BF16','soft_default',*[f'soft8_{i}' for i in range(7)]]
checks=cpu_checks=0;summary={}
for ci,c in enumerate(d['cases']):
 n=c['sequences'];assert c['case']==ci and len(c['variants'])==9 and len(c['phases'])==2
 assert [v['label'] for v in c['variants']]==labels
 for phase,ph in enumerate(c['phases']):
  assert ph['phase']==phase and ph['query_K_V_changed']==bool(phase) and ph['length_slots_changed']==bool(phase)
  assert ph['length']==c['initial_length']-7*phase and [v['label'] for v in ph['checks']]==labels
  for r in ph['checks']:
   assert all(r[k] for k in ('writer_storage_bitwise','padding_preserved','scratch_tail_preserved','RNG_unchanged'))
   x=r['reference'];assert x['passed'] and x['failed_elements']==0 and x['atol']==.02 and x['rtol']==.01 and math.isfinite(x['max_abs']);checks+=1
  for x in ph['cpu_reference_checks']+ph['cpu_soft_reference_checks']:
   assert x['passed'] and x['failed_elements']==0 and x['atol']==.02 and x['rtol']==.01 and math.isfinite(x['max_abs']);cpu_checks+=1
 for vi,v in enumerate(c['variants']):
  e=[e for e in d['events'] if e['case']==ci and e['label']==v['label']]
  assert [x['trial'] for x in e]==list(range(7)) and all(math.isfinite(x['ms']) and x['ms']>0 for x in e)
  assert statistics.median(x['ms'] for x in e)==v['median_ms']
  if vi==0:assert v['config'] is v['compiled_metadata'] is None;continue
  r=v['compiled_metadata'];assert r['IS_3D'] and r['sequences']==n and r['key_dtype']=='torch.uint8' and r['query_dtype']=='torch.bfloat16'
  assert r['stages']==1 and isinstance(r['n_regs'],int) and isinstance(r['n_spills'],int)
  if vi==1:
   assert v['config']==dict(default=True)
   selected=n in (8,16,32,40,48,56,64)
   assert r['tile']==(16 if selected else 32) and r['warps']==(1 if selected else 2 if 8<=n<=64 else 4)
   assert r['segments']==(8 if n==64 else 64 if n==1 else 4 if 16<=n<=64 else 16)
  else:
   cfg=PLANS[vi-2];assert v['config']==cfg and r['tile']==cfg['TILE_SIZE'] and r['warps']==cfg['num_warps'] and r['segments']==cfg['segments']
  assert r['grid']==[n//2+n,2,r['segments']]
 best=min(c['variants'][1:],key=lambda v:v['median_ms']);control=c['variants'][0]['median_ms'];old=c['variants'][1]['median_ms']
 summary[c['name']]=dict(BF16_ms=control,old_Soft8_ms=old,best=best,time_change_vs_BF16_percent=100*(best['median_ms']/control-1))
r=dict(complete=True,sources=109,cases=10,variants_per_case=9,changed_phases=20,recorded_full_output_checks=checks,CPU_head_reference_checks=cpu_checks,event_groups=630,writer_included=True,summary=summary,official_benchmark=False,adopted=False,scope='Recorded assertions and source/event audit; no full tensors serialized or model quality proof')
(root/'local_download_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='summary'}))
