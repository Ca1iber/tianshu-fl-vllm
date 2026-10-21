# SPDX-License-Identifier: Apache-2.0
"""Base471 checks retained; actual new JIT/graph/grid evidence supersedes base selector labels."""
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_BM8_full471_audit',BASE.parent/'tianshu_m_range_prefill_service_v2_20261021/audit_profile.py');parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_extension(root,state,result,require_C64=True):
 r=read(next((root/'bm8_decode_records').glob('binding_*.json')));selection=read(BASE/'selected_plan.json')
 assert r['installed'] and r['completed'] and r['mode']=='candidate' and r['new_path_enabled_during_capture'] and r['selected_sequences']==[64] and r['IS_3D_only'] and r['no_new_packed_weights'] and r['RNG_not_read_or_modified'] and r['other_batches_and_2D_delegate_original']
 for name in ('policy','selection','kernel'):assert r[name+'_sha256']==state['frozen_sources'][r[name+'_source']]
 rows=list(r['routes'].values());selected=[x for x in rows if x['selected']];captures=[x for x in selected if x['actually_inside_capture']]
 assert sum(x['calls'] for x in captures)>=42 and all(x['calls']>0 for x in rows)
 for x in selected:
  assert x['sequences']==64 and x['IS_3D'] and x['BLOCK_M']==8 and x['BLOCK_Q']==1 and x['TILE_SIZE']==16 and x['segments']==4 and x['actual_grid']==[64,2,4]
  assert x['actual_source_sha256']==selection['kernel_sha256'] and x['timed_component_binary_exact_match'] and x['binary_sha256']==selection['timed_component_kernels'][x['cache_hash']]['binary_sha256'] and x['n_spills']==0
 for x in rows:
  if not x['selected']:assert x['actual_grid']==x['original_grid'] and x['BLOCK_M']==16 and x['BLOCK_Q']==2 and x['actual_source_sha256']==r['base_pair_source_sha256']
 evidence=[]
 if require_C64:
  trace=read(next((root/'graph_segments_records').glob('trace_*.json')));context=state.get('workload',{}).get('input_tokens',1)
  for key,g in trace['graphs'].items():
   calls=sum(x['calls'] for x in captures if x['graph_key']==key)
   if calls>=42 and g['capture_calls']>=1 and g['replay_calls']>=100 and g['frames'].get('64',{}).get('max_computed',0)>=context:evidence.append(dict(graph_key=key,new_kernel_capture_calls=calls,replay_calls=g['replay_calls'],actual_C64_frames=g['frames']['64']))
  assert evidence,'No all42 new BM8 capture bound to actual C64 request Graph replay'
 old=read(BASE.parent/'tianshu_m_range_prefill_official_results_20261021/OFFICIAL_GATE_32g.json')['model_frozen_sources'];assert len(old)==471 and all(state['frozen_sources'][p]==h for p,h in old.items())
 result.update(BM8_record=r,BM8_real_capture_verified=True,BM8_actual_C64_request_graph_replay_verified=require_C64,BM8_C64_graph_evidence=evidence,original471_exact_subset=True,no_new_packed_weights=True,base_pair_selector_source_sha256=result.get('active_decode_source_sha256'),active_decode_source_sha256=selection['kernel_sha256'],actual_decode_source_scope='New source for actual C64 only; other batches use original pair_rows',legacy_pair_record_scope=r['old_pair_record_scope'],runtime_source_count=len(state['frozen_sources']))
 return result

def audit_runtime(root,state,official=False):
 result=parent.audit_runtime(root,state,official=official);return audit_extension(root,state,result,require_C64=True)
