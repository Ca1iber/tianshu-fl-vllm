# SPDX-License-Identifier: Apache-2.0
"""Keep full471 audits; explicitly add QKV64 integer and payload evidence."""
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_QKV_NN_full_471_audit',BASE.parent/'tianshu_m_range_prefill_service_v2_20261021/audit_profile.py');parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_extension(root,state,result,require_C64=True):
 r=read(next((root/'qkv_nn_decode_records').glob('route_*.json')));selection=read(BASE/'selected_plan.json');reference={i['prefix']:i for i in read(BASE.parent/'tianshu_qkv_prefill_int8_service_v2_20261021/CHECKPOINT_SIGNATURES_32g.json')['banks']}
 assert r['installed'] and r['completed'] and r['mode']=='candidate' and r['new_integer_M']==[64] and r['new_path_enabled_during_capture'] and r['all_42_M64_real_capture_routes_verified']
 assert r['other_M_delegates_original_method'] and r['original_BF16_weights_retained'] and r['warm_RNG_unchanged'] and r['warm_CPU_threads_restored'] and r['warm_before_actual_graph_capture']
 for key in ('policy','kernel','selection','checkpoint_signature'):assert r[key+'_sha256']==state['frozen_sources'][r[key+'_source']]
 assert len(r['banks'])==42 and {b['prefix'] for b in r['banks']}==set(reference) and sum(b['payload_bytes'] for b in r['banks'])==r['actual_extra_packed_payload_bytes']==220631040
 for b in r['banks']:
  assert b['signature']==reference[b['prefix']] and b['original_weight_unchanged'] and b['payload_bytes']==5253120 and b['physical_shape']==[2048,2560] and b['physical_stride']==[2560,1]
  assert b['inside_profiler']['active'] and b['inside_profiler']['sha256']==state['frozen_sources'][b['inside_profiler']['source']]
 assert len(r['warmups'])==1 and all(w['all_A_quantization_exact'] and w['selected_real_W_INT64_outputs_passed'] and w['component_binary_exact_match'] for w in r['warmups'])
 actual=[x for x in r['routes'].values() if x['implementation']=='NN_A8W8_M64'];captured=[x for x in actual if x['actually_inside_capture']]
 assert len({x['prefix'] for x in captured})==42
 assert all(x['M']==64 and x['input_stride']==[2048,1] and x['configuration']==selection['configuration'] and x['component_binary_exact_match'] and x['binary_sha256']==selection['expected_component_binary_sha256'] and x['kernel_sha256']==selection['kernel_sha256'] and x['calls']>0 for x in actual)
 old=read(BASE.parent/'tianshu_m_range_prefill_official_results_20261021/OFFICIAL_GATE_32g.json')['model_frozen_sources'];assert len(old)==471 and all(state['frozen_sources'][p]==h for p,h in old.items())
 dense=result['dense_int8_record'];assert r['actual_model_memory_usage_bytes']==dense['model_memory_usage_bytes'] and r['actual_model_memory_usage_bytes']+16*1024**2>=dense['parameter_device_bytes']+dense['packed_payload_bytes']+220631040
 qualified_graphs=[]
 if require_C64:
  assert result['actual_FL_Graph_capture_and_replay_verified']
  trace=read(next((root/'graph_segments_records').glob('trace_*.json')))
  context=state.get('workload',{}).get('input_tokens',1)
  for key,g in trace['graphs'].items():
   frame=g['frames'].get('64',{})
   if g['replay_calls']>=100 and frame.get('max_computed',-1)>=context:
    matching=[x for x in captured if x['graph_key']==key]
    if len({x['prefix'] for x in matching})==42:
     assert g['capture_calls']>=1 and g['graph_object_id'] is not None
     qualified_graphs.append(dict(graph_key=key,capture_calls=g['capture_calls'],replay_calls=g['replay_calls'],actual_C64_frames=frame,captured_QKV_prefixes=42))
  assert qualified_graphs,'No actual request C64 replay of all42 captured QKV NN kernels'
 result['QKV_actual_C64_graph_replay_evidence']=qualified_graphs
 result.update(qkv_NN_record=r,QKV_M64_integer_captured_all42=True,QKV_M64_capture_plus_qualified_graph_replay_verified=require_C64,original471_exact_subset=True,new_model_Math_required=True,extra_packed_weight_bytes=dense['packed_payload_bytes']+220631040,legacy_dense_payload_field_scope='Original43 banks only; actual native profiler also includes42 new QKV banks',runtime_source_count=len(state['frozen_sources']))
 return result

def audit_runtime(root,state,official=False):
 assert state['mode']=='candidate';result=parent.audit_runtime(root,state,official=official);return audit_extension(root,state,result,require_C64=True)
