# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_cold_full_dense_runtime_audit',BASE/'dense_audit.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==59 and state['profile_environment']['TIANSHU_DENSE_INT8_MODE']=='candidate'
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    dense=result['dense_int8_record'];assert dense['startup_mode']==state['mode']
    before,during,after=[dense[key] for key in ('cpu_threads_before_load','cpu_threads_during_load','cpu_threads_after_load')]
    assert before==after and during==(before if state['mode']=='native' else 4)
    assert all(row['packing_CPU_threads']==during and row['packing_method']==('GPU' if state['mode']=='gpu4' else 'CPU') for row in dense['banks'])
    reference={row['prefix']:row for row in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')}
    assert len(reference)==len(dense['banks'])==43
    for row in dense['banks']:
        expected=reference[row['prefix']]
        assert all(row[key]==expected[key] for key in ('shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes'))
    bindings=next((root/'dense_int8_records').glob('bindings_*.json'));worker_pid=int(bindings.stem.split('_')[-1])
    sql=read(root/f'cold_start_records/sql/reflection_{worker_pid}.json')
    assert sql['installed'] and sql['mode']==state['mode'] and sql['original_sha256']==state['frozen_sources'][sql['original_source']]
    assert not sql['preimported_module'] and sql['target_reflection']==(state['mode']!='native') and sql['total_calls']>0
    assert state['frozen_sources'][sql['hook_source']]==sql['hook_sha256'] and sql['original_class_other_methods_unchanged']
    result.update(cold_start_mode=state['mode'],full_59_environment_verified=True,packing_CPU_threads=during,CPU_threads_restored=True,
        all_43_banks_same_as_qualified_Math=True,loader_seconds=dense['load_model_seconds'],weight_packing_seconds=sum(row['packing_seconds'] for row in dense['banks']),
        actual_worker_SQL_record=sql,actual_worker_SQL_calls_verified=True,model_quality_not_newly_evaluated=True)
    return result
