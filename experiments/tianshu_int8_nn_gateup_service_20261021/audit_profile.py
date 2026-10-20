# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_nn_gateup_qualified_audit',BASE.parent/'tianshu_lm_precision_service_20261020/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==62
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    files=list((root/'int8_nn_gateup_records').glob('route_*.json'));assert len(files)==1;row=read(files[0])
    assert row['installed'] and row['completed'] and row['after_original_capture'] and row['mode']==state['mode']
    assert all(row[k] for k in ('original_parameter_and_checkpoint_unchanged','logical_packed_banks_unchanged','LM_route_unchanged','no_additional_packed_weight_payload','static_transpose_inside_DeviceMemoryProfiler','old_dense_binding_kernel_fields_identify_original_NT_module','actual_NN_execution_evidence_in_this_route_record','Python_calls_do_not_count_Graph_replays'))
    reference={r['prefix']:r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')};assert len(row['banks'])==42
    for bank in row['banks']:
        assert bank['logical_shape']==[12288,2048] and bank['logical_sha256']==reference[bank['prefix']]['packed_sha256'] and bank['payload_bytes']==12288*2048
        assert bank['logical_stride']==([1,12288] if state['mode']=='candidate' else [2048,1])
        assert bank['inside_profiler']['active'] and bank['inside_profiler']['sha256']==state['frozen_sources'][bank['inside_profiler']['source']]
        assert bank['full_device_transpose_equal']==(state['mode']=='candidate') and bank['full_logical_hash_equal'] and bank['original_BF16_weight_version_preserved']
    for name in ('policy','NT','NN'):assert row[name+'_sha256']==state['frozen_sources'][row[name+'_source']]
    for m in (8,64):
        routes=[r for r in row['routes'].values() if r['M']==m and r['actually_inside_capture']];assert len({r['prefix'] for r in routes})==42
        for route in routes:
            assert route['implementation']==('NN' if state['mode']=='candidate' else 'NT') and route['calls']>0
            assert route['kernel_sha256']==state['frozen_sources'][route['kernel_source']]
            if state['mode']=='candidate':assert route['configuration']==row['selected_configurations'][str(m)]
    result.update(mode=state['mode'],NN_gateup_record=row,all_42_logical_packed_banks_identical=True,physical_weight_layout_candidate=(state['mode']=='candidate'),LM_precision_and_route_unchanged=True,
        legacy_dense_kernel_fields_scope='Original NT module only; NN_gateup_record routes identify actual executed GateUp kernel/config/binary. Graph capture calls are not replay counts.',new_model_Math_not_run=True)
    return result
