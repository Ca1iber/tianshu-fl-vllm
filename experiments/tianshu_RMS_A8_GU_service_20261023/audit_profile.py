# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(root,state):
 paths=list((root/'RMS_A8_GU_records').glob('route_*.json'));assert len(paths)==1;r=read(paths[0]);assert r['installed'] and r['completed'] and r['mode']=='candidate' and r['all42_new_norm_capture_verified'] and r['warm_before_capture_RNG_unchanged'] and r['no_new_packed_weight_payload'] and r['no_global_tensor_workspace'] and r['QKV_Down_LM_routes_unchanged'] and r['old_compatibility_index_is_actual_GEMM_kernel_calls_not_old_custom_op_or_A8_calls']
 for name in ('source','norm','GEMM'):
  p=r[name] if name=='source' else r[name+'_source'];h=r['source_sha256'] if name=='source' else r[name+'_sha256'];assert state['frozen_sources'][p]==h==sha(p)
 numeric=read(BASE.parent/'tianshu_IR_RMS_A8_20261023/INDEPENDENT_CPU_GATE_32g.json');lead=read(BASE.parent/'tianshu_IR_RMS_A8_chain_20261023/INDEPENDENT_CPU_GATE_32g.json');assert numeric['complete'] and lead['complete'] and lead['component_lead'];assert r['expected_norm_binary']==numeric['compiled_candidate_kernels']['True']['binary_sha256'];old=read(next((root/'nn_silu_fx_records').glob('route_*.json')));assert r['expected_GEMM_binary']==old['component_expected_binary_sha256']
 captured=[v for v in r['routes'].values() if v['actually_inside_capture']];assert {v['bank_id'] for v in captured}==set(range(42));oldcaptured=[v for v in old['routes'].values() if v['actually_inside_capture'] and v['M']==64];assert len(oldcaptured)==len(captured)==42
 for row in r['routes'].values():
  assert row['M']==64 and row['K']==2048 and row['N']==12288 and row['calls']>0 and row['norm_component_binary_exact'] and row['GEMM_component_binary_exact'] and row['norm_binary_sha256']==r['expected_norm_binary'] and row['GEMM_binary_sha256']==r['expected_GEMM_binary'] and row['standalone_A8_quantizer_executed'] is False
  matched=[v for v in old['routes'].values() if v.get('wrapper_source')==r['source'] and v['bank_id']==row['bank_id'] and v['actually_inside_capture']==row['actually_inside_capture'] and v['graph_key']==row['graph_key']];assert len(matched)==1;v=matched[0];assert v['calls']==row['calls'] and v['wrapper_sha256']==r['source_sha256'] and v['standalone_A8_quantizer_executed'] is False and v['norm_and_A8_fused'] and v['binary_sha256']==row['GEMM_binary_sha256'] and v['kernel_sha256']==r['GEMM_sha256']
  if row['actually_inside_capture']:assert row['graph_key'] is not None
 for row in oldcaptured:assert row.get('wrapper_source')==r['source'] and row.get('standalone_A8_quantizer_executed') is False
 passes=[read(p) for p in (root/'RMS_A8_GU_records/fx_pass').glob('pass_*.json')];assert len(passes)==86;static=[p for p in passes if p['compile_range']==[64,64]];dynamic=[p for p in passes if p['compile_range']==[1,4096]];assert len(static)==len(dynamic)==43 and sorted(b for p in static for b in p['bank_ids'])==list(range(42)) and sum(p['replaced'] for p in static)==42
 assert all(p['replaced']==0 and p['before_sha256']==p['after_sha256'] and p['all_other_ranges_unchanged'] for p in dynamic)
 for p in passes:assert state['frozen_sources'][p['source']]==p['source_sha256']==sha(p['source'])
 return dict(RMS_A8_GU_record=r,new_RMS_A8_GU_all42_static_M64_capture_and_real_binaries_verified=True,new_RMS_A8_GU_no_fictitious_old_quantizer_calls=True,new_RMS_A8_GU_dynamic_ranges_unchanged=True)
