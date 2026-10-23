# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit_startup(root,state):
 files=list((root/'prefill_RMS_QKV_records').glob('route_*.json'));assert len(files)==1;r=read(files[0]);assert r['installed'] and r['completed'] and r['mode']=='candidate' and r['all42_static4096_real_warm_invocations_verified'] and r['original_compile_sizes']==[64] and r['actual_compile_sizes']==r['actual_compile_sizes_at_capture']==[64,4096] and r['no_new_packed_weight_payload'] and r['no_global_tensor_workspace'] and r['QKV_other_M_O_Down_GU_LM_routes_unchanged'] and r['compatibility_index_only_actual_same_native_MM_and_epilogue_not_old_custom_op_or_A8'] and r['Python_calls_are_not_Graph_replays']
 for name in ('source','norm','epilogue','parent_binding'):
  path=r[name] if name=='source' else r[name+'_source'];expected=r['source_sha256'] if name=='source' else r[name+'_sha256'];assert state['frozen_sources'][path]==expected==sha(path)
 gate=read(r['component_gate']);assert state['frozen_sources'][r['component_gate']]==r['component_gate_sha256']==sha(r['component_gate']);assert gate['complete'] and gate['component_lead'];assert r['expected_norm_binaries']=={str(has):gate['compiled'][f'fused_norm_M4096_residual_{has}']['binary_sha256'] for has in (False,True)} and r['expected_epilogue_binary']==gate['compiled']['QKV_dequant_M4096']['binary_sha256']
 assert r['profile_probe_CPU_GPU_RNG_unchanged'] and {p['bank_id'] for p in r['profile_probes']}=={0,2}
 for probe in r['profile_probes']:
  context=probe['context'];assert probe['explicit_zero_probe_not_real_request'] and probe['zero_output'] and probe['original_weights_versions_preserved'] and context['native_memory_context_active'] and context['manual_KV_bytes'] is None and state['frozen_sources'][context['source']]==context['sha256']==sha(context['source']);assert probe['M']==4096 and probe['K']==2048 and probe['N']==2560 and probe['has_residual']==(probe['bank_id']==2) and probe['integer_workspace_bytes']==4096*2560*4 and probe['peak_after']>=probe['peak_before']>=probe['allocated_before']
 passes=[read(path) for path in (root/'prefill_RMS_QKV_records/fx_pass').glob('pass_*.json')];assert len(passes)==129;ranges={tuple(p['compile_range']) for p in passes};assert ranges=={(1,4096),(64,64),(4096,4096)}
 for interval in ranges:
  selected=[p for p in passes if tuple(p['compile_range'])==interval];assert len(selected)==43
  if interval==(4096,4096):assert sorted(bank for p in selected for bank in p['bank_ids'])==list(range(0,84,2)) and sum(p['replaced'] for p in selected)==42 and sum(sum(p['has_residual']) for p in selected)==41
  else:assert all(p['replaced']==0 and p['before_sha256']==p['after_sha256'] and p['all_other_ranges_unchanged'] for p in selected)
 for p in passes:
  assert state['frozen_sources'][p['source']]==p['source_sha256']==sha(p['source']);name=f"pass_{files[0].stem.removeprefix('route_')}_{p['index']:03d}_{p['compile_range'][0]}_{p['compile_range'][1]}.py";assert sha(root/'prefill_RMS_QKV_records/fx_pass'/name)==p['after_sha256']
 real=[v for v in r['routes'].values() if not v['explicit_zero_profile_probe']];assert {v['bank_id'] for v in real}==set(range(0,84,2))
 return dict(prefill_RMS_QKV_record=r,new_prefill_RMS_QKV_startup_all42_static4096_verified=True,new_prefill_RMS_QKV_memory_peak_explicit_probes_accounted=True,new_prefill_RMS_QKV_actual_compile_sizes=[64,4096],old_compile_sizes_unchanged_flags_scope='earlier original hook only; new overall config is [64,4096]',new_prefill_RMS_QKV_general_and_M64_graphs_unchanged=True)

def audit(root,state):
 result=audit_startup(root,state);r=result['prefill_RMS_QKV_record'];parent=read(next((root/'native_attention_records').glob('route_*.json')));expected_keys=set();runtime=[]
 for key,row in r['routes'].items():
  assert row['M']==4096 and row['K']==2048 and row['N']==2560 and row['bank_id'] in range(0,84,2) and row['has_residual']==(row['bank_id']!=0) and row['calls']>0 and row['standalone_A8_quantizer_executed'] is False and row['norm_A8_fused'] and row['actual_original_weight_and_NN_pack_pointers_verified'] and row['norm_component_binary_exact'] and row['epilogue_component_binary_exact'] and row['norm_binary_sha256']==r['expected_norm_binaries'][str(row['has_residual'])] and row['epilogue_binary_sha256']==r['expected_epilogue_binary']
  oldkey='actual_native_MM_from_prefill_RMS_QKV:'+key
  if row['explicit_zero_profile_probe']:assert row['phase']=='profile_probe' and oldkey not in parent['routes'];continue
  expected_keys.add(oldkey);old=parent['routes'][oldkey];assert old['bank_id']==row['bank_id'] and old['calls']==row['calls'] and old['phase']==row['phase'] and old['capturing']==row['capturing'] and old['graph_key']==row['graph_key'] and old['wrapper_source']==r['source'] and old['wrapper_sha256']==r['source_sha256'] and old['standalone_A8_quantizer_executed'] is False and old['norm_A8_fused'] and old['binary_sha256']==row['epilogue_binary_sha256'] and old['implementation']=='native_A8W8_attention_projection' and old['INT32_workspace_bytes']==4096*2560*4
  if row['phase']=='runtime' and not row['capturing']:runtime.append(row)
 assert {key for key,value in parent['routes'].items() if value.get('wrapper_source')==r['source']}==expected_keys;actual=sorted({row['bank_id'] for row in runtime})
 if state.get('quality_batch')==64 or state.get('stage')=='official':assert actual==list(range(0,84,2))
 result.update(new_prefill_RMS_QKV_runtime_bank_ids=actual,new_prefill_RMS_QKV_runtime_banks=len(actual),new_prefill_RMS_QKV_old_custom_op_and_standalone_A8_calls_not_claimed=True,new_prefill_RMS_QKV_same_native_MM_epilogue_and_original_packs_verified=True,new_prefill_RMS_QKV_no_global_activation_workspace=True);return result
