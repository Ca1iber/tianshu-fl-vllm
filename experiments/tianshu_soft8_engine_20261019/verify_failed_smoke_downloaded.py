# SPDX-License-Identifier: Apache-2.0
"""Scoped audit: byte allocation/profile succeeded, graph capture OOM, no API proof."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
d=json.loads((root/'result.json').read_text());assert d['phase']=='failed' and d['error']=="AssertionError('Server exited during startup')"
assert not d['short_4k_rounds'] and not d['seeded_canaries'] and len(d['profile_environment'])==40
assert 'Triton Error [CUDA]: out of memory' in (root/'server.log').read_text(errors='replace')
repo=Path(__file__).resolve().parents[2]
for name,h in d['frozen_sources'].items():
 assert hashlib.sha256((root/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==h
 if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==h
assert len(d['frozen_sources'])==134
files=list((root/'soft8_engine_records').glob('engine_*.json'));assert len(files)==1;e=json.loads(files[0].read_text())
assert e['enabled'] and e['mode']=='candidate' and e['specs']==[dict(layers=42,dtypes=['torch.uint8'],before_page_bytes=[16384],after_page_bytes=[8192])]
assert 'determine_available_memory' in e['reservation_stack'] and e['profile_calls']==1
r=e['scratch_reservation'];assert r['capacity_pages']==32768 and r['scratch_bytes']==32769*16384 and r['table_bytes']==32768*4
assert r['allocated_after']-r['allocated_before']==r['scratch_bytes']+r['table_bytes']
assert len(e['cache_views'])==1 and len(e['cache_views'][0])==42
for v in e['cache_views'][0]:assert v['dtype']=='torch.uint8' and v['shape']==[63773,2,16,2,128] and v['strides'][1:]==[4096,256,128,1]
routes=list(e['producer_records'].values());assert {r['sequences'] for r in routes}=={248,256}
assert all(not r['IS_3D'] and r['segments']==1 and r['tile']==128 and r['warps']==16 and r['n_regs']==256 and r['n_spills']==908 for r in routes)
proof=e['codec_proof'];original=Path(d['output_root'])
for name,h in ((proof['writer_path'],proof['writer_sha256']),(proof['reader']['generated_source'],proof['reader']['generated_sha256'])):
 assert hashlib.sha256((root/Path(name).relative_to(original)).read_bytes()).hexdigest()==h
assert 'No running processes found' in (root/'hardware_after.txt').read_text()
r=dict(complete=True,failed_original_preserved=True,source_snapshots=134,actual_UInt8_spec_layers=42,actual_UInt8_cache_view_layers=42,KV_capacity_tokens=1020368,arena_allocation_inside_memory_profile=True,API_requests=0,Math_not_run=True,formal_benchmark_not_run=True,OOM_at_large_batch_graph_capture=True,spill_is_observed_not_proven_exclusive_cause=True,actual_installed_runner_file_in_original_snapshot='/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py' in d['frozen_sources'],adopted=False)
(root/'local_failure_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
