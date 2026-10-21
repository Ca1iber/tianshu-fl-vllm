# SPDX-License-Identifier: Apache-2.0
"""Retain every qualified450 check; replace only actual prefill route evidence."""
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def read(path):return json.loads(Path(path).read_text())
legacy=load('_range_qualified450_audit',BASE.parent/'tianshu_nn_silu_fx_screen_20261021/audit_profile.py')
adapted=load('_range_actual_prefill_audit',BASE/'prefill_audit.py');adapted.parent=legacy.prefill.parent;legacy.prefill=adapted

def audit_runtime(root,state,official=False):
 mode=state['mode'];assert mode in ('control','candidate')
 result=legacy.audit_runtime(root,dict(state,mode='candidate'),official=official)
 files=list((root/'m_range_prefill_records').glob('route_*.json'));assert len(files)==1;record=read(files[0])
 assert record['mode']==mode and record['installed'] and record['completed'] and record['active']
 assert record['no_additional_packed_weight_payload'] and record['precision_M_whitelist_not_expanded'] and record['M64_existing_fusion_preserved'] and record['compile_sizes_unchanged']
 assert record['warm_RNG_unchanged'] and record['CPU_threads_restored'] and record['activated_after_qualified_capture_and_warm'] and record['fallback_activation_warm_before_capture_RNG_unchanged']
 for key in ('policy','kernel','activation','quantizer','selection'):assert record[key+'_sha256']==state['frozen_sources'][record[key+'_source']]
 selection=read(BASE/'selected_plan.json');component=read(selection['component_gate']);assert record['configuration_by_M']==selection['configuration_by_M']
 assert record['component_gate_sha256']==selection['component_gate_sha256']==state['frozen_sources'][selection['component_gate']]
 rows=[read(path) for path in (root/'m_range_prefill_records/fx_pass').glob('pass_*.json')];assert len(rows)==86
 dynamic=[row for row in rows if row['compile_range']==[1,4096]];static=[row for row in rows if row['compile_range']==[64,64]];assert len(dynamic)==len(static)==43
 assert sorted(i for row in dynamic for i in row['bank_ids'])==list(range(42)) and all(row['replaced']==0 and row['before_sha256']==row['after_sha256'] for row in static)
 assert sum(row['replaced'] for row in dynamic)==(42 if mode=='candidate' else 0)
 if mode=='control':assert all(row['before_sha256']==row['after_sha256'] for row in dynamic) and not record['routes'] and not record['warmups']
 else:
  assert record['fallback_activation_warm_performed'] and len(record['warmups'])==len(record['configuration_by_M'])
  assert all(row['all_A_quantization_exact'] and row['CPU_selected_real_W_INT64_activation_passed'] and row['component_binary_exact_match'] for row in record['warmups'])
  assert set(record['actual_kernel_by_M'])==set(record['configuration_by_M'])
  for m,row in record['actual_kernel_by_M'].items():assert row['component_binary_exact_match'] and row['binary_sha256']==component['actual_timed_candidate_kernels'][m]['binary_sha256']
  fused=[row for row in record['routes'].values() if row['implementation']=='M_range_two_dot_fused' and row['stage']=='active' and not row['capturing']]
  assert len({row['prefix'] for row in fused if row['M']==4096})==42
  assert all(row['configuration']==record['configuration_by_M'][str(row['M'])] and row['input_stride']==[2048,1] and row['calls']>0 for row in fused)
 original=read(BASE.parent/'tianshu_nn_silu_fx_c64_quality_20261021/C64_QUALITY_GATE_32g.json')['model_frozen_sources'];assert len(original)==450 and all(state['frozen_sources'][p]==h for p,h in original.items())
 result.update(mode=mode,range_prefill_record=record,qualified450_same_in_both_arms=True,original450_source_map_exact_subset=True,full_inherited_attention_KV_scheduler_weights_LM_checks_retained=True,native_dynamic_graphs_preserved=(mode=='control'),legacy_native_dynamic_graphs_preserved_scope='Old M64 pass only; this new pass records and audits the dynamic linear/activation replacements',legacy_M8_decode_unchanged_scope='Old M64 pass only; new dynamic opaque fallback delegates original linear and uses same generated activation arithmetic',new_model_Math_required=True)
 return result
