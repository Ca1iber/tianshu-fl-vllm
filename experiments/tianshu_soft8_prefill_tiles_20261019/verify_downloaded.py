# SPDX-License-Identifier: Apache-2.0
"""Audit completed component records and frozen sources; no model quality claim."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from plans import PLANS
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
d=json.loads((root/'result.json').read_text());assert d['complete'] and d['phase']=='complete'
assert d['writer_included'] and not d['official_benchmark'] and not d['adopted'] and d['plans']==PLANS
assert len(d['profile_environment'])==38 and len(d['source_snapshots'])==85
repo=Path(__file__).resolve().parents[2]
for name,h in d['source_snapshots'].items():
 copy=root/'source_snapshot'/Path(name).relative_to('/');assert hashlib.sha256(copy.read_bytes()).hexdigest()==h,name
 if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
  assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==h,name
prerequisite=json.loads((Path(__file__).parent/'PREREQUISITE_32g.json').read_text());assert prerequisite['complete']
assert all(d['source_snapshots'][name]==h for name,h in prerequisite['frozen_sources'].items())
assert len(d['events'])==224 and len(d['cases'])==2
labels=['BF16',*[f'soft8_{i}' for i in range(15)]];summary={};checks=0;cpu_checks=0
for ci,c in enumerate(d['cases']):
 assert c['case']==ci and c['name']==['prefill2048','chunk2048'][ci] and c['initial_length']==[2048,4096][ci] and c['strided']==bool(ci)
 assert len(c['phases'])==2 and len(c['variants'])==16
 for phase,ph in enumerate(c['phases']):
  assert ph['phase']==phase and ph['query_K_V_changed']==bool(phase)
  assert ph['length']==c['initial_length']-(7 if phase and ci else 0)
  assert ph['length_slots_changed']==bool(phase and ci) and [x['label'] for x in ph['checks']]==labels
  for check in ph['checks']:
   assert all(check[k] for k in ('writer_storage_bitwise','padding_preserved','scratch_untouched','RNG_unchanged'))
   r=check['reference'];assert r['passed'] and r['failed_elements']==0 and r['atol']==.02 and r['rtol']==.01 and math.isfinite(r['max_abs']);checks+=1
  for r in ph['cpu_reference_checks']+ph['cpu_soft_reference_checks']:
   assert r['passed'] and r['failed_elements']==0 and r['atol']==.02 and r['rtol']==.01 and math.isfinite(r['max_abs']);cpu_checks+=1
 assert [v['label'] for v in c['variants']]==labels
 for vi,v in enumerate(c['variants']):
  events=[e for e in d['events'] if e['case']==ci and e['label']==v['label']]
  assert [e['trial'] for e in events]==list(range(7)) and all(math.isfinite(e['ms']) and e['ms']>0 for e in events)
  assert statistics.median(e['ms'] for e in events)==v['median_ms']
  if vi==0:assert v['config'] is None and v['compiled_metadata'] is None;continue
  cfg=PLANS[vi-1];m=v['compiled_metadata'];assert v['config']==cfg and not m['IS_3D'] and m['sequences']==m['segments']==1
  assert m['BLOCK_M']==cfg['BLOCK_M'] and m['BLOCK_Q']==cfg['BLOCK_M']//8 and m['tile']==cfg['TILE_SIZE']
  assert m['warps']==cfg['num_warps'] and m['stages']==cfg['num_stages'] and m['grid']==[2048//m['BLOCK_Q']+1,2]
  assert m['key_dtype']=='torch.uint8' and m['query_dtype']=='torch.bfloat16' and isinstance(m['n_regs'],int) and isinstance(m['n_spills'],int)
 best=min(c['variants'][1:],key=lambda v:v['median_ms']);control=c['variants'][0]['median_ms']
 summary[c['name']]=dict(BF16_ms=control,best=best,slowdown=best['median_ms']/control)
r=dict(complete=True,source_snapshots=85,cases=2,variants_per_case=16,changed_phases=4,recorded_full_output_checks=checks,CPU_head_reference_checks=cpu_checks,event_groups=224,writer_included=True,summary=summary,official_benchmark=False,adopted=False,scope='Audit of recorded GPU assertions, CPU reference records, events and sources; full tensors not serialized, no engine or model quality proof')
(root/'local_download_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
