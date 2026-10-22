# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def audit(root,state):
 files=list((root/'native_attention_records').glob('route_*.json'));assert len(files)==1;record=read(files[0]);selection=read(BASE/'selected_plan.json');expected={b['prefix']:b for b in selection['banks']};assert record['installed'] and record['completed'] and record['capture_completed'] and record['phase']=='runtime' and record['mode']==state['attention_mode'];assert record['one_NN_layout_per_bank'] and record['original_BF16_retained'] and record['no_global_tensor_workspace'] and record['warm_CPU_GPU_RNG_unchanged'] and record['load_complete'] and record['rows_by_role']==selection['rows_by_role']
 for kind in ('source','dense','epilogue','packer','selection'):
  path=record[kind] if kind=='source' else record[kind+'_source'];digest=record['source_sha256'] if kind=='source' else record[kind+'_sha256'];assert state['frozen_sources'][path]==digest==hashlib.sha256(Path(path).read_bytes()).hexdigest()
 assert len(record['banks'])==84 and {b['prefix'] for b in record['banks']}==set(expected);payload=0
 for bank in record['banks']:
  proof=expected[bank['prefix']];n=proof['N'];assert bank['role']==proof['role'] and bank['packed_shape']==[2048,n] and bank['packed_stride']==[n,1] and bank['original_weight_version_preserved'];assert all(bank[k]==proof[k] for k in ('original_BF16_sha256','packed_sha256','scales_sha256'));prof=bank['inside_DeviceMemoryProfiler'];assert prof['active'] and prof['sha256']==state['frozen_sources'][prof['source']];assert bank['payload_bytes']==2048*n+4*n;payload+=bank['payload_bytes']
 assert payload==selection['packed_payload_bytes']==record['packed_payload_bytes'] and record['model_memory_usage_bytes']>=payload;keys=set(selection['expected_dequant_binaries']);assert set(record['native_kernels'])==keys and {f'{r["role"]}:{r["M"]}' for r in record['warmups']}==keys and len(record['warmups'])==6
 for row in record['warmups']:
  key=f'{row["role"]}:{row["M"]}';prof=row['inside_DeviceMemoryProfiler'];assert row['zero_output'] and prof['active'] and prof['sha256']==state['frozen_sources'][prof['source']] and row['INT32_workspace_bytes']==row['M']*row['N']*4 and row['binary_sha256']==selection['expected_dequant_binaries'][key]
 for key,value in record['native_kernels'].items():assert value['component_binary_exact'] and value['binary_sha256']==selection['expected_dequant_binaries'][key]
 routes=list(record['routes'].values());capture=[r for r in routes if r['phase']=='capture' and r['capturing'] and r['M']==64];assert len({r['bank_id'] for r in capture})==84 and all(r['implementation']==('native_A8W8_attention_projection' if record['mode']=='candidate' and r['role']=='QKV' else 'qualified_BF16_vendor_mm') for r in capture)
 for module in record['BF16_vendor_modules'].values():
  assert module['actual_vendor_mm_direct'] and module['source_sha256']=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090';assert state['frozen_sources'][module['source']]==module['source_sha256'] and state['frozen_sources'][module['launch_source']]==module['launch_source_sha256']
 assert record['BF16_vendor_modules'];hits=[r for r in routes if r['phase']=='runtime' and not r['capturing'] and r['implementation']=='native_A8W8_attention_projection']
 for r in routes:
  assert r['calls']>0 and r['K']==2048 and r['N']==(2560 if r['role']=='QKV' else 2048)
  if r['implementation']=='native_A8W8_attention_projection':assert record['mode']=='candidate' and r['M'] in selection['rows_by_role'][r['role']] and r['input_stride']==[2048,1] and r['binary_sha256']==selection['expected_dequant_binaries'][f'{r["role"]}:{r["M"]}'] and r['INT32_workspace_bytes']==r['M']*r['N']*4
  else:assert r['implementation']=='qualified_BF16_vendor_mm'
 if record['mode']=='candidate':
  profile=[r for r in routes if r['phase']=='profile' and r['M']==4096 and r['implementation']=='native_A8W8_attention_projection'];assert len({r['bank_id'] for r in profile})==84 and any(w['context'] and w['context']['native_memory_context_active'] and not w['context']['manual_KV_bytes'] and w['native_calls_after']-w['native_calls_before']>=84 for w in record['profile_windows'])
 else:assert not hits and all(r['implementation']=='qualified_BF16_vendor_mm' for r in routes)
 by_role={role:dict(banks=len({r['bank_id'] for r in hits if r['role']==role}),M_values=sorted({r['M'] for r in hits if r['role']==role}),M4096_banks=len({r['bank_id'] for r in hits if r['role']==role and r['M']==4096})) for role in ('QKV','O')}
 if state.get('quality_batch')==64 and record['mode']=='candidate':assert all(row['banks']==row['M4096_banks']==42 for row in by_role.values())
 return dict(native_attention_record=record,new_attention_memory_profile_all84_accounted=True,new_attention_packed_payload_bytes=payload,actual_attention_runtime_native_by_role=by_role,actual_attention_M64_capture_BF16_banks=(42 if record['mode']=='candidate' else 84),actual_attention_M64_capture_native_banks=(42 if record['mode']=='candidate' else 0),new_attention_Python_calls_not_claimed_as_replays=True)

def audit_startup(root,state):
 files=list((root/'native_attention_records').glob('route_*.json'));assert len(files)==1;r=read(files[0]);assert r['completed'] and r['capture_completed'] and r['phase']=='runtime' and len(r['banks'])==84 and len(r['warmups'])==6;capture=[v for v in r['routes'].values() if v['phase']=='capture' and v['capturing'] and v['M']==64];assert len({v['bank_id'] for v in capture})==84 and all(v['implementation']==('native_A8W8_attention_projection' if r['mode']=='candidate' and v['role']=='QKV' else 'qualified_BF16_vendor_mm') for v in capture)
 return dict(new_attention_startup_all84_banks=True,new_attention_M64_QKV_native_O_BF16_policy_verified=True)
