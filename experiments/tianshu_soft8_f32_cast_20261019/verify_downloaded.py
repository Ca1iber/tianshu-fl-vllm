# SPDX-License-Identifier: Apache-2.0
"""Audit three-path component evidence; model quality and service remain separate."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from source import encode_bits,render_reader,render_writer,prior,WRITER_SHA
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=read(root/'result.json');assert d['complete'] and d['phase']=='complete' and not d['official_benchmark'] and not d['adoption_allowed']
assert d['variant_order']==['BF16','Soft8','Soft8F32'] and len(d['cases'])==21 and len(d['profile_environment'])==38
assert d['actual_builder_and_KV_writer_checked'] and d['changed_Graph_replays_verified'] and d['real_spec_layers']==42
assert d['all65536_GPU_encoded_bytes']==[encode_bits(i) for i in range(65536)]
expected=[(((c&127)<<4)+14848 if(c&127)>=8 else 0)|((c&128)<<8) for c in range(256)]
assert d['all256_GPU_decoded_bits']==d['all256_GPU_f32_decoded_bits']==expected
repo=Path(__file__).resolve().parents[2]
for name,h in d['frozen_sources'].items():
 assert sha(root/'source_snapshot'/Path(name).relative_to('/'))==h
 if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):assert sha(repo/Path(name).relative_to('/workspace/vllm-plugin-FL'))==h
pre=read(root/'preflight_sources.json');assert pre['frozen_sources']==d['frozen_sources'] and not pre['complete'] and pre['codec_proof']==d['codec_proof']
proof=d['codec_proof'];remote=Path(proof['writer_path']).parents[1]
def artifact(path):return root/Path(path).relative_to(remote)
cv=next(k for k,v in d['frozen_sources'].items() if v=='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289')
raw=(root/'source_snapshot'/Path(cv).relative_to('/')).read_bytes()
for key,renderer in (('reader',prior.render_source),('f32_reader',render_reader)):
 text,meta=renderer(raw);item=proof[key];assert artifact(item['generated_source']).read_text()==text and sha(artifact(item['generated_source']))==item['generated_sha256']==meta['generated_sha256']
wp=next(k for k,v in d['frozen_sources'].items() if v==WRITER_SHA)
assert artifact(proof['writer_path']).read_text()==render_writer((root/'source_snapshot'/Path(wp).relative_to('/')).read_bytes())
assert sha(artifact(proof['writer_path']))==proof['writer_sha256']
checks=cpu_checks=events=dense_checks=dense_elements=bitwise=0;timings={}
names=['prefill32','prefill33','prefill2048','chunk2048','ragged','mixed64','mixed_ragged','decode64','decode64_5k','decode64_65','decode1_33','decode1_4k','decode80','decode128','decode256','bounded_fallback','decode8','decode16','decode24','decode32','unreordered']
assert [c['name'] for c in d['cases']]==names
for c in d['cases']:
 assert c['query_and_KV_and_slots_and_lengths_changed_in_place'] and len(c['phases'])==2
 assert [v['layout'] for v in c['cache_layouts']]==d['variant_order']
 for ph in c['phases']:
  assert all(ph[k] for k in ('KV_writer_all_storage_bitwise_exact','output_padding_preserved','segment_scratch_tail_preserved','same_graph_RNG_unchanged'))
  assert len(ph['reference_checks'])==3
  for r in ph['reference_checks']+ph['cpu_checks']+ph['cpu_soft8_checks']:
   assert r['passed'] and r['failed_elements']==0 and r['atol']==.02 and r['rtol']==.01 and math.isfinite(r['max_abs']);checks+=1
  bitwise+=bool(ph['f32_vs_byte_bitwise']);assert math.isfinite(ph['f32_vs_byte_max_abs']) and ph['f32_vs_byte_max_abs']>=0
  if ph['f32_vs_byte_bitwise']:assert ph['f32_vs_byte_max_abs']==0
 dc=c['dense_prefill_checks']
 if c['name'].startswith('decode') or c['name']=='bounded_fallback':assert dc==[]
 else:
  assert [(x['phase'],x['layout']) for x in dc]==[(ph,v) for ph in (0,1) for v in ('Soft8','Soft8F32')]
  nd=next((i for i,q in enumerate(c['query_lengths']) if q!=1),len(c['query_lengths']))
  for r in dc:
   assert r['active_cells_bitwise'] and r['scratch_guard_preserved'] and r['distinct_dense_page_ids']
   assert r['checked_elements']==sum(c['phases'][r['phase']]['kv_lengths'][nd:])*512
   dense_checks+=1;dense_elements+=r['checked_elements']
 assert len(c['raw_ms'])==3 and all(len(v)==7 and all(math.isfinite(t) and t>0 for t in v) for v in c['raw_ms'])
 med=list(map(statistics.median,c['raw_ms']));assert c['median_ms']==med;events+=21
 assert abs(c['time_change_percent']-100*(med[1]/med[0]-1))<1e-10
 assert abs(c['f32_time_change_vs_byte_percent']-100*(med[2]/med[1]-1))<1e-10
 timings[c['name']]=dict(BF16_ms=med[0],byte_ms=med[1],f32_cast_ms=med[2],f32_vs_byte_percent=c['f32_time_change_vs_byte_percent'])
assert events==441 and dense_checks==32
routes=list(d['actual_soft8_route_records'].values());assert {r['codec_variant'] for r in routes}=={'Soft8','Soft8F32'}
for r in routes:
 assert r['key_dtype']=='torch.uint8' and r['query_dtype']=='torch.bfloat16' and isinstance(r['n_regs'],int) and isinstance(r['n_spills'],int)
 want=proof['reader' if r['codec_variant']=='Soft8' else 'f32_reader']['generated_source'];assert r['kernel_source']==want
 if r['pure_decode']:assert r['BLOCK_M']==16 and r['BLOCK_Q']==2 and r['tile']==16 and r['warps']==1
reservation=d['scratch_reservation'];assert reservation['capacity_pages']==32768 and reservation['scratch_bytes']==32769*16384
assert len({v['arena_ptr'] for v in d['dense_prefill_metadata'].values()})==1
r=dict(complete=True,component_only=True,cases=21,variants=3,changed_phases=42,F32_vs_byte_bitwise_phases=bitwise,recorded_reference_checks=checks,event_groups=events,dense_checks=dense_checks,dense_elements=dense_elements,frozen_sources=len(d['frozen_sources']),timings=timings,official_benchmark=False,fresh_Math_not_run=True,adopted=False,scope='Source/event and recorded full GPU checks; tensors not serialized, no full model proof')
(root/'local_download_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
