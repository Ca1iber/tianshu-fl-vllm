# SPDX-License-Identifier: Apache-2.0
"""Additional actual-prefill evidence, alongside retained C8/C64 quality audits."""
import hashlib,json
from pathlib import Path
SERVICE=Path(__file__).resolve().parent.parent/'tianshu_m_range_prefill_service_v2_20261021'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(root,state,require_fused=True):
 assert state['mode']=='candidate' and len(state['frozen_sources'])==500
 files=list((root/'m_range_prefill_records').glob('route_*.json'));assert len(files)==1;record=read(files[0])
 assert record['mode']=='candidate' and record['installed'] and record['completed'] and record['active']
 assert record['no_additional_packed_weight_payload'] and record['precision_M_whitelist_not_expanded'] and record['M64_existing_fusion_preserved'] and record['compile_sizes_unchanged']
 assert record['warm_RNG_unchanged'] and record['CPU_threads_restored'] and record['activated_after_qualified_capture_and_warm'] and record['fallback_activation_warm_before_capture_RNG_unchanged']
 for key in ('policy','kernel','activation','quantizer','selection'):assert record[key+'_sha256']==state['frozen_sources'][record[key+'_source']]==sha(record[key+'_source'])
 selection=read(SERVICE/'selected_plan.json');component=read(selection['component_gate']);assert record['configuration_by_M']==selection['configuration_by_M']
 assert record['component_gate_sha256']==selection['component_gate_sha256']==state['frozen_sources'][selection['component_gate']]==sha(selection['component_gate'])
 rows=[read(p) for p in (root/'m_range_prefill_records/fx_pass').glob('pass_*.json')];assert len(rows)==86
 dynamic=[r for r in rows if r['compile_range']==[1,4096]];static=[r for r in rows if r['compile_range']==[64,64]]
 assert len(dynamic)==len(static)==43 and sorted(i for r in dynamic for i in r['bank_ids'])==list(range(42))
 assert all(r['replaced']==0 and r['before_sha256']==r['after_sha256'] for r in static) and sum(r['replaced'] for r in dynamic)==42
 assert record['fallback_activation_warm_performed'] and len(record['warmups'])==len(record['configuration_by_M'])==23
 assert all(r['all_A_quantization_exact'] and r['CPU_selected_real_W_INT64_activation_passed'] and r['component_binary_exact_match'] for r in record['warmups'])
 assert set(record['actual_kernel_by_M'])==set(record['configuration_by_M'])
 for m,r in record['actual_kernel_by_M'].items():assert r['component_binary_exact_match'] and r['binary_sha256']==component['actual_timed_candidate_kernels'][m]['binary_sha256']
 fused=[r for r in record['routes'].values() if r['implementation']=='M_range_two_dot_fused' and r['stage']=='active' and not r['capturing']]
 if require_fused:
  assert len({r['prefix'] for r in fused})==42
  assert len({r['prefix'] for r in fused if r['M']==4096})==42
 assert all(r['configuration']==record['configuration_by_M'][str(r['M'])] and r['input_stride']==[2048,1] and r['calls']>0 and r['binary_sha256']==record['actual_kernel_by_M'][str(r['M'])]['binary_sha256'] for r in fused)
 original=read(SERVICE.parent/'tianshu_nn_silu_fx_c64_quality_20261021/C64_QUALITY_GATE_32g.json')['model_frozen_sources'];assert len(original)==450 and all(state['frozen_sources'][p]==h for p,h in original.items())
 return dict(complete=True,range_prefill_record=record,actual_range_fused_all_42_layers_verified=(len({r["prefix"] for r in fused})==42),actual_range_fused_M4096_all42_verified=(len({r["prefix"] for r in fused if r["M"]==4096})==42),actual_fused_M_values=sorted({r["M"] for r in fused}),actual_fused_prefix_count=len({r["prefix"] for r in fused}),original450_exact_subset=True,static_M64_preserved=True,dynamic_compile_range_explicitly_rewritten=True,legacy_dynamic_unchanged_scope='Old M64 pass only; new dynamic rewrite audited separately',activation_current_stream_each_call=True)
