# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;COMPONENT=EXP/'tianshu_native_dotsaddz_down42_20261023'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def audit_startup(root,state):
 files=list((root/'dots_Down64_records').glob('route_*.json'));assert len(files)==1;r=read(files[0]);assert r['complete'] and r['capture_completed'] and r['phase']=='runtime' and r['load_complete'] and r['load_RNG_unchanged'];assert r['original_checkpoint_not_modified'] and r['original_BF16_and_W8_retained'] and r['no_global_tensor_workspace'] and r['Python_calls_are_not_graph_replays']
 for path,h in [(r['source'],r['source_sha256']),(r['selection_source'],r['selection_sha256']),(r['component_gate'],r['component_gate_sha256'])]:assert state['frozen_sources'][path]==h==sha(path)
 selection=read(BASE/'SELECTED_BANKS_32g.json');cpu=read(COMPONENT/'CPU_GATE_32g.json');assert selection['CPU_gate_sha256']==sha(COMPONENT/'CPU_GATE_32g.json');expected={b['layer']:b for b in selection['banks']};assert len(r['banks'])==42 and {b['layer'] for b in r['banks']}==set(range(42));payload=0;ids=set()
 for b in r['banks']:
  e=expected[b['layer']];assert b['bank_id'] not in ids;ids.add(b['bank_id']);assert b['prefix']==f"model.layers.{b['layer']}.mlp.down_proj" and b['original_sha256']==e['original_sha256'] and b['original_weight_version_preserved'];assert b['packed_file']==e['file'] and b['packed_file_sha256']==e['file_sha256']==sha(e['file']);prof=b['inside_DeviceMemoryProfiler'];assert prof['active'] and prof['function']=='load_model' and state['frozen_sources'][prof['source']]==prof['source_sha256']==sha(prof['source']);assert len(b['entries'])==3
  for row in b['entries']:
   shape,dtype,key={'w':([6144,256],'torch.int32','pack_sha256'),'s':([48,2048],'torch.bfloat16','scale_sha256'),'z':([48,2048],'torch.bfloat16','zero_sha256')}[row['name']];assert row['shape']==shape and row['dtype']==dtype and row['stride']==[shape[1],1] and row['sha256']==e[key];assert row['bytes']==shape[0]*shape[1]*(4 if row['name']=='w' else 2)
  assert b['payload_bytes']==sum(v['bytes'] for v in b['entries']);payload+=b['payload_bytes']
 assert payload==r['additional_packed_payload_bytes']==42*(6144*256*4+2*48*2048*2);assert r['own_buffers_in_actual_model_accounting'] and r['model_memory_usage_bytes']+16*1024**2>=r['all_model_unique_parameter_buffer_storage_bytes']>=payload
 assert len(r['warmups'])==len(r['profile_probes'])==42 and {w['bank_id'] for w in r['warmups']}=={p['bank_id'] for p in r['profile_probes']}==ids
 for row in r['warmups']:
  p=row['inside_DeviceMemoryProfiler'];assert row['M']==64 and row['zero_output'] and p['active'] and p['function']=='load_model' and state['frozen_sources'][p['source']]==p['source_sha256']
 for row in r['profile_probes']:
  p=row['inside_native_worker_memory_profile'];assert row['M']==64 and row['zero_output'] and row['explicit_scratch_probe_not_main_M4096_path'] and p['active'] and p['function']=='determine_available_memory' and p['manual_KV_bytes'] is None and state['frozen_sources'][p['source']]==p['source_sha256']
 assert len(r['profile_windows'])==1
 for w in r['profile_windows']:assert w['context']['active'] and w['context']['manual_KV_bytes'] is None and w['RNG_unchanged'] and w['peak_after']>=w['peak_before'] and w['peak_after']>=w['allocated_after']
 assert len(r['libraries'])==3 and all(cpu['sources'][p]==h==sha(p) for p,h in r['libraries'].items())
 captures=[v for v in r['routes'].values() if v['phase']=='capture' and v['capturing']];assert {v['bank_id'] for v in captures}==ids
 for v in r['routes'].values():assert v['implementation']=='native_W4A16_dotsaddz' and v['M']==64 and v['K']==6144 and v['N']==2048 and v['input_stride']==[6144,1] and v['bank_id'] in ids and v['calls']>0
 old=read(next((root/'native_down_records').glob('route_*.json')));assert not [v for v in old['routes'].values() if v['M']==64 and v['phase']=='capture' and v['capturing']]
 return dict(dots_Down64_record=r,dots_Down64_startup_all42_captured=True,dots_Down64_load_and_native_memory_scratch_all42_accounted=True,dots_Down64_additional_packed_payload_bytes=payload,dots_Down64_original_BF16_capture_requirement_explicitly_replaced=True,dots_Down64_original_checkpoint_and_other_precision_preserved=True,actual_Down64_W4_capture_banks=42,old_Down64_BF16_capture_banks=0,no_Python_call_counters_claimed_as_replays=True)
def audit(root,state,inherited):
 result=audit_startup(root,state);r=result['dots_Down64_record'];hits=[v for v in r['routes'].values() if v['phase']=='runtime' and not v['capturing']];result.update(dots_Down64_runtime_Python_banks=len({v['bank_id'] for v in hits}),dots_Down64_captured_replay_linked_to_inherited_model_M64_execution=(state['quality_batch']==64 and inherited['actual_C64_fused_replay_verified']))
 if state['quality_batch']==64:assert result['dots_Down64_captured_replay_linked_to_inherited_model_M64_execution'] and inherited['verified_dense_batch']==64 and inherited['LM_head_precision_in_verified_batch']=='INT8'
 return result
