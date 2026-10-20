# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
from base_runtime_audit import audit_runtime as baseline_audit
BASE=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit_runtime(root,state,official=False):
    result=baseline_audit(root,state,official=official)
    if state.get('workload',{}).get('input_tokens')==4096:assert result['zero_preemptions']
    rows=list((root/'long_segments_records').glob('bindings_*.json'))
    assert len(rows)==1
    record=read(rows[0]);expected='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106' if state['mode']=='candidate' else 'dc69f525d8c7686a8977f50e901af39820ce92486894890631c53fc01dc46700'
    assert record['installed'] and record['mode']==state['mode'] and record['actual_kernel_sha256']==expected
    assert sha(record['producer_source'])==sha(record['reducer_source'])==expected
    assert record['policy']==read(BASE/'batch_policy.json') and record['policy_sha256']==sha(BASE/'batch_policy.json')
    assert record['RNG_not_read_or_modified'] and record['unchanged_reader_python_dispatch'] and record['KV_format_writer_and_model_parameters_unchanged']
    files=list((root/'graph_segments_records').glob('trace_*.json'));assert len(files)==1
    trace=read(files[0]);assert set(trace['bound'])=={'graph','runner','routing'}
    for name,digest in trace['sources'].items():assert state['frozen_sources'][name]==digest and sha(name)==digest
    assert trace['attention_arguments_and_return_values_forwarded'] and trace['RNG_not_read_or_modified']
    captured=[x for x in trace['launches'].values() if x['actually_inside_capture'] and x['graph_key'] is not None]
    assert captured
    selected_captures=0
    for key,graph in trace['graphs'].items():
        if graph['replay_calls']:
            assert graph['capture_calls']>=1 and graph['graph_object_id'] is not None
            assert sum(x['calls'] for x in captured if x['graph_key']==key)>=42
    for row in captured:
        cfg=row['configuration']
        if cfg['IS_3D'] and 33<=cfg['sequences']<=56:
            assert cfg['segments']==(16 if state['mode']=='candidate' else 4)
            selected_captures+=row['calls']
    assert selected_captures>=42
    frames=[(int(n),v) for graph in trace['graphs'].values() for n,v in graph['frames'].items()]
    assert any(n>=(64 if official else 1) and v['max_computed']>=state.get('workload',{}).get('input_tokens',1) for n,v in frames)
    result.update(actual_FL_Graph_capture_and_replay_verified=True,selected_attention_captures=selected_captures,
                  graph_count=len(trace['graphs']),active_kernel_sha256=expected,mode=state['mode'])
    return result
