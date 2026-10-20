# SPDX-License-Identifier: Apache-2.0
"""Full 56-key environment plus unchanged paired-K baseline and true INT8 calls."""
import hashlib,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_dense_baseline_pair_audit',BASE/'baseline_pair_audit.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==57 and state['profile_environment']['TIANSHU_PAIR_LOAD_MODE']=='candidate'
    assert state['profile_environment']['TIANSHU_DENSE_INT8_MODE']==state['mode']
    assert state['profile_environment']['VLLM_DISABLE_COMPILE_CACHE']=='1'
    observed=read(root/'observed_server_environment.json')
    assert observed['environment']==state['profile_environment'] and observed['command']==state['server_command']
    # The inherited attention/memory policy remains candidate in BOTH arms;
    # our mode controls dense integer GEMM only. The adapted copy checks all
    # actual 56 environment keys, without changing or filtering observations.
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    files=list((root/'dense_int8_records').glob('bindings_*.json'));assert len(files)==1
    row=read(files[0]);assert row['installed'] and row['mode']==state['mode'] and row['load_complete'] and row['compile_cache_disabled']
    for key in ('policy','kernel'):
        assert sha(row[key+'_source'])==state['frozen_sources'][row[key+'_source']]==row[key+'_sha256']
    for path,digest in row['native_method_sources'].items():assert sha(path)==state['frozen_sources'][path]==digest
    assert len(row['visited'])==43 and sum(x['role']=='gateup' for x in row['visited'])==42 and sum(x['role']=='lm_head' for x in row['visited'])==1
    for item in row['visited']:
        evidence=item['inside_profiler'];assert evidence['profiler_not_exited'] and evidence['function']=='load_model'
        assert sha(evidence['source'])==state['frozen_sources'][evidence['source']]==evidence['sha256']
    assert row['original_BF16_weights_retained'] and row['checkpoint_never_modified'] and row['embedding_not_packed']
    assert row['packing_inside_actual_active_DeviceMemoryProfiler'] and row['RNG_not_read_or_modified']
    assert row['model_memory_usage_bytes']+16*1024**2>=row['parameter_device_bytes']+row['packed_payload_bytes']
    if state['mode']=='candidate':
        assert len(row['banks'])==row['actual_packed_banks']==43 and row['packed_payload_bytes']==1326938112
        assert len({x['prefix'] for x in row['banks']})==43 and sum(x['packed_bytes'] for x in row['banks'])==1326938112
        assert all(x['original_weight_unchanged'] and x['inside_profiler']['profiler_not_exited'] for x in row['banks'])
        trace=read(next((root/'graph_segments_records').glob('trace_*.json')))
        for batch,context,replays in ((64,4096,100),(8,128,1)):
            selected=set()
            for launch in row['launches'].values():
                if launch['role']!='gateup' or launch['M']!=batch or not launch['actually_inside_capture'] or launch['graph_key'] is None:continue
                graph=trace['graphs'][launch['graph_key']]
                if graph['capture_calls']>=1 and graph['replay_calls']>=replays and any(int(n)==batch and frame['max_computed']>=context for n,frame in graph['frames'].items()):selected.add(launch['bank_id'])
            assert len(selected)==42,(batch,len(selected))
            lm=[r for r in row['launches'].values() if r['role']=='lm_head' and r['M']==batch and not r['actually_inside_capture']]
            assert sum(r['calls'] for r in lm)>=replays,(batch,lm)
        assert all(r['kernel_sha256']==row['kernel_sha256'] and r['M'] in (8,64) and r['calls']>0 for r in row['launches'].values())
    else:assert row['banks']==[] and row['launches']=={} and row['actual_packed_banks']==row['packed_payload_bytes']==0
    result.update(mode=state['mode'],paired_K_mode='candidate',dense_int8_binding_verified=True,
        dense_int8_mode=state['mode'],dense_int8_record=row,extra_packed_weight_bytes=row['packed_payload_bytes'],
        actual_model_loader_profiler_active_during_packing_verified=True,candidate_C8_and_C64_calls_verified=state['mode']=='candidate',
        dense_INT8_model_quality_not_yet_evaluated=True)
    return result
