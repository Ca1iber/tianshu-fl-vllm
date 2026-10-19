# SPDX-License-Identifier: Apache-2.0
"""Audit real encoded writer/forward records; this does not create a model gate."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from source import encode_bits,render_writer,WRITER_SHA,prior
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=read(root/'result.json');assert d['phase']=='complete' and d['complete'] and not d['official_benchmark'] and not d['adoption_allowed']
assert d['actual_builder_and_KV_writer_checked'] and d['changed_Graph_replays_verified'] and d['quantization_changes_values'] and d['real_spec_layers']==42
assert d['all65536_GPU_encoded_bytes']==[encode_bits(i) for i in range(65536)]
assert d['all256_GPU_decoded_bits']==[(((i&127)<<4)+14848 if(i&127)>=8 else 0)|((i&128)<<8) for i in range(256)]
repo=Path(__file__).resolve().parents[2]
for name,h in d['frozen_sources'].items():
 assert sha(root/'source_snapshot'/Path(name).relative_to('/'))==h
 if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):assert sha(repo/Path(name).relative_to('/workspace/vllm-plugin-FL'))==h
assert len(d['profile_environment'])==38
fx=d['actual_allocator_CPU_fixtures'];assert len(fx)==4
for r in fx:
 assert r['shape']==[3,2,16,2,128] and r['view_aliases_actual_raw']
 want=16384 if r['layout']=='BF16' else 8192
 assert r['page_bytes']==want and r['stride_bytes']==want+(256 if r['packed'] else 0)
 assert r['dtype']==('torch.bfloat16' if r['layout']=='BF16' else 'torch.uint8')
assert {(r['layout'],r['packed']) for r in fx}=={(s,p) for s in ('BF16','Soft8') for p in (False,True)}
writer_path=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py')
original=(root/'source_snapshot'/writer_path.relative_to('/')).read_bytes();assert hashlib.sha256(original).hexdigest()==WRITER_SHA
actual_writer=root/Path(d['codec_proof']['writer_path']).relative_to(Path(d['codec_proof']['writer_path']).parents[1])
assert actual_writer.read_text()==render_writer(original)
assert sha(actual_writer)==d['codec_proof']['writer_sha256']
reader_proof=d['codec_proof']['reader'];inputs=[name for name,h in d['frozen_sources'].items() if h==reader_proof['original_sha256']]
assert inputs
reader_text,proof=prior.render_source((root/'source_snapshot'/Path(inputs[0]).relative_to('/')).read_bytes())
remote_root=Path(d['codec_proof']['writer_path']).parents[1]
reader_file=root/Path(reader_proof['generated_source']).relative_to(remote_root)
assert reader_file.read_text()==reader_text and sha(reader_file)==reader_proof['generated_sha256']

names=['prefill32','prefill33','prefill2048','chunk2048','ragged','mixed64','mixed_ragged','decode64','decode64_5k','decode64_65','decode1_33','decode8','decode16','decode24','decode32','unreordered']
assert len(d['cases'])==16 and [r['name'] for r in d['cases']]==names
checks=0;event_count=0;timings={}
for c in d['cases']:
 assert c['query_and_KV_and_slots_and_lengths_changed_in_place'] and len(c['phases'])==2
 assert [r['layout'] for r in c['cache_layouts']]==['BF16','Soft8']
 block=16384 if c['strided'] else 8192
 assert all(r['strides']==[block,4096,256,128,1] for r in c['cache_layouts'])
 for phase,ph in enumerate(c['phases']):
  assert ph['phase']==phase
  assert all(ph[k] for k in ('KV_writer_all_storage_bitwise_exact','output_padding_preserved','segment_scratch_tail_preserved','same_graph_RNG_unchanged'))
  assert len(ph['reference_checks'])==2
  for r in [*ph['reference_checks'],*ph['cpu_checks'],*ph['cpu_soft8_checks']]:
   assert r['passed'] and r['failed_elements']==0 and r['atol']==.02 and r['rtol']==.01
   assert math.isfinite(r['max_abs']) and r['max_abs']>=0;checks+=1
  app=ph['approximation_to_BF16'];assert math.isfinite(app['max_abs']) and math.isfinite(app['relative_RMS'])
 assert len(c['raw_ms'])==2 and all(len(a)==7 for a in c['raw_ms'])
 assert all(t>0 for a in c['raw_ms'] for t in a)
 med=list(map(statistics.median,c['raw_ms']));assert med==c['median_ms'];event_count+=14
 assert abs(100*(med[1]/med[0]-1)-c['time_change_percent'])<1e-10
 timings[c['name']]=dict(BF16_ms=med[0],Soft8_pipeline_ms=med[1],time_change_percent=c['time_change_percent'])
routes=list(d['actual_soft8_route_records'].values());assert routes
assert any(r['IS_3D'] and r['sequences']==64 and r['segments']==8 and r['tile']==16 and r['warps']==1 for r in routes)
assert any(not r['IS_3D'] and r['tile']==128 and r['warps']==16 for r in routes)
assert all(r['key_dtype']=='torch.uint8' and r['query_dtype']=='torch.bfloat16' and r['stages']==1 for r in routes)
counts=d['Python_route_counts_not_graph_replay_counts'];assert all(counts[k]>0 for k in ('encoded_writer','decode','prefill','mixed'))
r=dict(complete=True,source_snapshots=len(d['frozen_sources']),actual_GPU_encoded_values=65536,actual_GPU_decoded_bits=256,actual_spec_CPU_fixtures=4,actual_builder_writer_forward_cases=16,changed_graph_phases=32,recorded_reference_checks=checks,event_groups=event_count,writer_cost_included=True,timings=timings,
 scope='Recorded full GPU checks, exhaustive serialized codec values, sources and events audited; no full engine allocator, model service or fresh Math proof',official_benchmark=False,adopted=False)
(root/'local_download_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
