# SPDX-License-Identifier: Apache-2.0
"""Recompute controlled-event medians and audit unchanged physical-address sources."""
import argparse,hashlib,json,statistics
from pathlib import Path
from source import ATTENTION_SHA,WRITER_SHA,CHUNKS,render_attention,render_writer
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=read(root/'result.json');assert d['phase']=='complete' and d['complete'] and not d['official_benchmark'] and not d['adopted']
assert d['all_outputs_control_bitwise'] and d['changed_Graph_replays_verified'] and d['full_physical_storage_writer_verified']
assert len(d['profile_environment'])==38 and d['same_CV_L0_attention_arithmetic'] and d['V_layout_unchanged'] and d['page16_BF16']
repo=Path(__file__).resolve().parents[2]
for name,h in d['frozen_sources'].items():
 assert sha(root/'source_snapshot'/Path(name).relative_to('/'))==h
 if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):assert sha(repo/Path(name).relative_to('/workspace/vllm-plugin-FL'))==h
remote=Path(d['generated_sources'].__iter__().__next__()).parent.parent
for name,h in d['generated_sources'].items():assert sha(root/Path(name).relative_to(remote))==h
raw_a=(root/'generated/_tianshu_kblock_attention_0.py').read_bytes();raw_w=(root/'generated/_tianshu_kblock_writer_0.py').read_bytes()
assert hashlib.sha256(raw_a).hexdigest()==ATTENTION_SHA and hashlib.sha256(raw_w).hexdigest()==WRITER_SHA
for x in CHUNKS:
 assert (root/f'generated/_tianshu_kblock_attention_{x}.py').read_text()==render_attention(raw_a,x)
 assert (root/f'generated/_tianshu_kblock_writer_{x}.py').read_text()==render_writer(raw_w,x)
names=['decode64_4k','decode64_5k','decode32_ragged','decode1_33','prefill33','prefill2048','chunk2048','ragged_prefill']
assert len(d['cases'])==8 and [c['name'] for c in d['cases']]==names and len(d['events'])==280
checks=0;timings={}
for case,c in enumerate(d['cases']):
 assert c['case']==case and c['writer_included_in_events'] and len(c['phases'])==2
 for phase,ph in enumerate(c['phases']):
  assert ph['phase']==phase and ph['query_K_V_changed']==bool(phase)
  assert ph['lengths_changed']==(ph['lengths']!=c['initial_lengths']) and ph['slot_mapping_changed']==(bool(phase) and ph['lengths_changed'])
  assert [r['x'] for r in ph['checks']]==[0,*CHUNKS]
  for r in ph['checks']:
   assert all(r[k] for k in ('control_bitwise','full_physical_writer_exact','output_padding_preserved','segment_scratch_tail_preserved','RNG_unchanged'))
   assert r['reference']['passed'] and r['reference']['bad']==0 and r['reference']['atol']==.02 and r['reference']['rtol']==.01;checks+=1
  assert ph['CPU_reference_checks'] and all(r['passed'] and r['bad']==0 for r in ph['CPU_reference_checks'])
 for x in (0,*CHUNKS):
  es=[e for e in d['events'] if (e['case'],e['x'])==(case,x)];assert len(es)==7 and {e['trial'] for e in es}==set(range(7))
  assert all(e['ms']>0 for e in es) and statistics.median(e['ms'] for e in es)==c['medians_ms'][str(x)]
  m=c['actual_attention_metadata'][str(x)]
  if c['name'].startswith('decode'):
   n=len(c['queries']);selected=n in (8,16,32,40,48,56,64)
   assert m['IS_3D'] and m['tile']==(16 if selected else 32) and m['num_warps']==(1 if selected else 4)
   assert m['segments']==(8 if n==64 else 64 if n==1 else 4)
  else:assert not m['IS_3D'] and m['tile']==128 and m['num_warps']==16 and m['segments']==1
  assert m['num_stages']==1
 timings[c['name']]=c['medians_ms']
r=dict(complete=True,frozen_sources=len(d['frozen_sources']),generated_sources=len(d['generated_sources']),cases=8,changed_graph_phases=16,numerical_fixture_checks=checks,event_records=280,
 control_bitwise_all_cases=True,V_layout_and_attention_math_unchanged=True,scope='Recorded component checks and frozen/generated sources; full GPU tensors not serialized; no model/backend/scheduler or quality proof',timings_ms=timings,official_benchmark=False,adopted=False)
(root/'local_download_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
