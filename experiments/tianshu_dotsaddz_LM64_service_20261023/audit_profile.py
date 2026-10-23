# SPDX-License-Identifier: Apache-2.0
"""Explicit LM64 W8-to-W4 replacement checks, never synthesizes parent counters."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def audit_startup(root,state):
 files=list((root/'dots_LM64_records').glob('route_*.json'));assert len(files)==1;r=read(files[0]);selection=read(BASE/'SELECTED_LM_32g.json')
 assert r['complete'] and r['capture_completed'] and r['phase']=='runtime' and r['load_complete'] and r['load_RNG_unchanged'] and r['CUDA_route_registered']
 for name in ('source','selection_source'):assert sha(r[name])==state['frozen_sources'][r[name]]==r[{'source':'source_sha256','selection_source':'selection_sha256'}[name]]
 assert r['actual_original_op_name']=='tianshu_dense_w8a8::linear' and r['genuine_original_bank_closure_bound'] and r['public_register_kernel_used'] and r['no_new_apply_override'] and r['original_CUDA_backend_retained_for_other_M_and_GU'] and r['original_checkpoint_and_BF16_W8_retained'] and r['no_global_tensor_workspace']
 assert r['original_init_source']==str(EXP/'tianshu_lm_precision_service_20261020/policy.py') and sha(r['original_init_source'])==state['frozen_sources'][r['original_init_source']]==r['original_init_sha256']
 assert len(r['banks'])==len(r['warmups'])==len(r['profile_probes'])==1 and r['additional_packed_payload_bytes']==selection['payload_bytes']==142049280 and r['own_buffers_in_actual_model_accounting'] and r['model_memory_usage_bytes']+16*1024**2>=r['all_model_unique_parameter_buffer_storage_bytes']
 b=r['banks'][0];assert b['original_BF16_sha256']==selection['W8_signature']['original_BF16_sha256'] and b['original_weight_version_preserved'] and sum(v['bytes'] for v in b['entries'])==142049280
 for entry in b['entries']:assert entry['sha256']==selection['buffers'][entry['name']]['array_sha256'] and entry['bytes']==selection['buffers'][entry['name']]['bytes']
 for item in [b,*r['warmups']]:
  c=item['inside_DeviceMemoryProfiler'];assert c['active'] and c['function']=='load_model' and sha(c['source'])==state['frozen_sources'][c['source']]==c['source_sha256']
 for item in r['profile_probes']:
  c=item['inside_native_worker_memory_profile'];assert c['active'] and c['function']=='determine_available_memory' and c['manual_KV_bytes'] is None and item['zero_output'] and sha(c['source'])==state['frozen_sources'][c['source']]==c['source_sha256']
 assert r['profile_windows'] and all(w['RNG_unchanged'] and w['context']['active'] and w['context']['manual_KV_bytes'] is None for w in r['profile_windows'])
 for route in r['routes'].values():assert route['M']==64 and route['K']==2048 and route['N']==130560 and route['bank_id']==b['bank_id'] and route['input_stride']==[2048,1] and route['implementation']=='native_W4A16_dotsaddz' and route['actual_parent_bank_identity_matched'] and route['calls']>0
 return dict(new_LM64_actual_W4_bank_verified=True,new_LM64_payload_bytes=142049280,new_LM64_load_and_native_peak_accounting_verified=True,new_LM64_parent_op_schema_and_bank_binding_verified=True,LM64_W4_record=r)
def audit_runtime(root,state,minimum_calls=0):
 result=audit_startup(root,state);r=result['LM64_W4_record'];calls=sum(v['calls'] for v in r['routes'].values() if v['phase']=='runtime' and not v['capturing']);assert calls>=minimum_calls
 old=read(next((root/'dense_int8_records').glob('bindings_*.json')));assert not [v for v in old['launches'].values() if v['role']=='lm_head' and v['M']==64]
 result.update(new_LM64_W4_actual_runtime_calls=calls,new_LM64_W4_calls_are_not_claimed_as_Graph_replays=True,old_LM64_W8_calls_absent=True,LM64_old_W8_guard_replaced_explicitly=True)
 return result
