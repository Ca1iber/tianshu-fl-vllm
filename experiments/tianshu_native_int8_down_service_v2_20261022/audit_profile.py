# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text())
def audit(root,state):
 files=list((root/'native_down_records').glob('route_*.json'));assert len(files)==1;record=read(files[0]);assert record['installed'] and record['completed'] and record['capture_completed'] and record['phase']=='runtime';assert record['mode']==state['down_mode'] and record['scope']==state['down_scope'];assert record['one_new_Down_NN_layout_only'] and record['original_BF16_retained'] and record['no_global_tensor_workspace'] and record['warm_CPU_GPU_RNG_unchanged'] and record['load_complete']
 for kind in ('source','quantizer','epilogue','packer','dense','selection'):
  path=record[kind] if kind=='source' else record[kind+'_source'];digest=record['source_sha256'] if kind=='source' else record[kind+'_sha256'];assert state['frozen_sources'][path]==digest==hashlib.sha256(Path(path).read_bytes()).hexdigest()
 selection=read(BASE/'selected_plan.json');expected={r['prefix']:r for r in selection['banks']};assert len(record['banks'])==42 and {b['prefix'] for b in record['banks']}==set(expected)
 payload=0
 for bank in record['banks']:
  assert bank['packed_shape']==[6144,2048] and bank['packed_stride']==[2048,1] and bank['original_weight_version_preserved'];assert all(bank[k]==expected[bank['prefix']][k] for k in ('original_BF16_sha256','packed_sha256','scales_sha256'));prof=bank['inside_DeviceMemoryProfiler'];assert prof['active'] and prof['sha256']==state['frozen_sources'][prof['source']];assert bank['payload_bytes']==6144*2048+2048*4;payload+=bank['payload_bytes']
 assert payload==record['packed_payload_bytes'] and record['model_memory_usage_bytes']>=payload
 assert len(record['warmups'])==4 and {r['M'] for r in record['warmups']}=={64,128,512,4096} and set(record['native_kernels'])=={'64','128','512','4096'}
 for row in record['warmups']:
  assert row['zero_output'] and row['INT32_workspace_bytes']==row['M']*2048*4 and row['inside_DeviceMemoryProfiler']['active'] and row['inside_DeviceMemoryProfiler']['sha256']==state['frozen_sources'][row['inside_DeviceMemoryProfiler']['source']] and row['binary_sha256']==selection['expected_dequant_binaries'][str(row['M'])]
 for m,row in record['native_kernels'].items():assert row['component_binary_exact_match'] and row['binary_sha256']==selection['expected_dequant_binaries'][m]
 supported=[64,128,512,4096] if record['scope']=='all' else [128,512,4096];assert record['supported_rows']==supported
 hits=[r for r in record['routes'].values() if r['phase']=='runtime' and not r['capturing'] and r['implementation']=='native_A8W8_Down'];capture=[r for r in record['routes'].values() if r['phase']=='capture' and r['capturing'] and r['M']==64]
 assert len({r['bank_id'] for r in capture})==42
 for row in record['routes'].values():
  assert row['calls']>0 and row['K']==6144 and row['N']==2048
  if row['implementation']=='native_A8W8_Down':assert record['mode']=='candidate' and row['M'] in supported and row['input_stride']==[6144,1] and row['binary_sha256']==selection['expected_dequant_binaries'][str(row['M'])] and row['INT32_workspace_bytes']==row['M']*2048*4
  else:assert row['implementation']=='original_BF16_Down'
 if record['mode']=='candidate':
  profile=[r for r in record['routes'].values() if r['phase']=='profile' and r['M']==4096 and r['implementation']=='native_A8W8_Down'];assert len({r['bank_id'] for r in profile})==42;assert any(w['context'] and w['context']['native_memory_context_active'] and not w['context']['manual_KV_bytes'] and w['native_calls_after']-w['native_calls_before']>=42 for w in record['profile_windows'])
  assert all(r['implementation']==('native_A8W8_Down' if record['scope']=='all' else 'original_BF16_Down') for r in capture)
  if state.get('quality_batch')==64:assert len({r['bank_id'] for r in hits if r['M']==4096})==42
 else:assert not hits and all(r['implementation']=='original_BF16_Down' for r in record['routes'].values())
 return dict(native_down_record=record,actual_native_Down_runtime_banks=len({r['bank_id'] for r in hits}),actual_native_Down_runtime_M_values=sorted({r['M'] for r in hits}),actual_Down_M64_capture_banks=len({r['bank_id'] for r in capture if r['implementation']=='native_A8W8_Down'}),new_Down_packed_payload_bytes=payload,new_Down_memory_profile_all42_accounted=True,no_claim_Python_counters_are_graph_replays=True)

def audit_startup(root,state):
    records=[read(p) for p in (root/'prefill_gemm_records').glob('mm_*.json')]
    assert {tuple(r['shape']) for r in records}=={(2048,2048,2048),(2048,12288,2048),(2048,2048,6144)}
    row=read(next((root/'native_down_records').glob('route_*.json')))
    assert row['completed'] and row['capture_completed'] and row['phase']=='runtime' and row['BF16_vendor_modules']
    for module in row['BF16_vendor_modules'].values():
        assert module['actual_vendor_mm_direct'] and module['source_sha256']=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
        assert module['source_sha256'] in state['frozen_sources'].values()
        assert module['launch_source_sha256']==state['frozen_sources'][module['launch_source']]
    return dict(original_three_vendor_GEMM_warm_shapes_restored=True,BF16_vendor_mm_direct_verified=True)
