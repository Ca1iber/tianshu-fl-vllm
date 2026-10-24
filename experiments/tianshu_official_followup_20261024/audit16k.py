# SPDX-License-Identifier: Apache-2.0
"""Audit the full model and actual 16K request batches, including final KV source."""
import collections,re
from pathlib import Path
PAIR='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
SCALAR='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
BM8='5cc0553c18a6aa3696b6bfb7e2f9433c45469cf4b4a1bb5eef806c5c1b5510a9'

def one(root,folder,pattern,read):
    files=list((root/folder).glob(pattern));assert len(files)==1,(folder,len(files))
    return read(files[0])

def verify(root,state,parent,r5,read,digest):
    assert state['benchmark_case']==[16384,1024,64,128]
    result=parent.audit_model(root,state,runtime=False)
    trace=one(root,'graph_segments_records','trace_*.json',read)
    pair=one(root,'pair_load_records','binding_*.json',read)
    assert pair['mode']=='candidate' and pair['parent_sha256']==SCALAR and pair['candidate_sha256']==PAIR
    assert digest(pair['parent_source'])==SCALAR and digest(pair['candidate_source'])==PAIR
    children=list((root/'bm8_decode_records').glob('binding_*.json'))
    assert len(children)==(1 if state['profile_name']=='bm8' else 0)
    child=read(children[0]) if children else None
    if child:assert child['completed'] and child['kernel_sha256']==BM8 and digest(child['kernel_source'])==BM8
    evidence=[];actual_frames=collections.Counter();allowed_lm_rows=set()
    for key,g in trace['graphs'].items():
        match=re.search(r'num_tokens=(\d+), num_reqs=(\d+), uniform=True, has_lora=False',g['descriptor'])
        if not match or g['runtime_mode']!='FULL' or g['capture_calls']!=1 or g['replay_calls']<100:continue
        tokens,padded=map(int,match.groups());assert tokens==padded and 1<=padded<=64
        frames={int(n):f for n,f in g['frames'].items() if f['count']>=10 and f['max_computed']>=16384}
        if not frames:continue
        base_rows=[r for r in pair['launches'].values() if r['graph_key']==key and r['inside_capture'] and r['IS_3D']]
        assert sum(r['calls'] for r in base_rows)>=42
        final_rows=([r for r in child['routes'].values() if r['graph_key']==key and r['actually_inside_capture'] and r['IS_3D']] if child else [])
        if final_rows:
            assert sum(r['calls'] for r in final_rows)>=42
            for row in final_rows:
                assert row['sequences']==padded and row['calls']>0
                assert row['actual_source_sha256'] in (PAIR,BM8)
                if row['selected']:
                    assert row['actual_source_sha256']==BM8 and row['timed_component_machine_code_exact_match'] and row['n_spills']==0
                else:assert row['actual_source_sha256']==PAIR and row['actual_grid']==row['original_grid']
            sources=sorted({r['actual_source_sha256'] for r in final_rows})
            capture_calls=sum(r['calls'] for r in final_rows)
        else:
            assert all(r['sequences']==padded and r['active_source_sha256'] in (PAIR,SCALAR) for r in base_rows)
            sources=sorted({r['active_source_sha256'] for r in base_rows})
            capture_calls=sum(r['calls'] for r in base_rows)
        assert len(sources)==1
        for n,f in frames.items():
            assert 1<=n<=padded
            actual_frames[n]+=f['count'];allowed_lm_rows.add(n)
        allowed_lm_rows.add(padded)
        evidence.append(dict(graph_key=key,graph_padded_requests=padded,replay_calls=g['replay_calls'],
            actual_frames=frames,all42_final_KV_capture_calls=capture_calls,final_KV_source_sha256=sources[0]))
    assert evidence and sum(actual_frames.values())>=100,'No actual long-context request Graph evidence'
    lm=one(root,'dots_LM64_records','route_*.json',read)
    calls=collections.Counter()
    for row in lm['routes'].values():
        if row['phase']=='runtime' and not row['capturing']:calls[int(row['M'])]+=row['calls']
    matched=[m for m in allowed_lm_rows if calls[m]>=10]
    assert matched,dict(actual_frames=actual_frames,LM_calls=calls)
    batch=max(matched,key=lambda m:calls[m])
    lm_runtime=r5.lm.audit_runtime(root,state,minimum_calls=10,batch=batch)
    future=one(root,'future_kv_records','*.json',read)
    assert future['mode']=='candidate' and future['tracked_requests']==future['outstanding_blocks']==0
    result.update(runtime=True,official16K_actual_batches_audited=True,
        actual_request_frame_counts=dict(actual_frames),LM_verified_runtime_matrix_rows=batch,
        LM_verified_runtime_calls=lm_runtime['new_LM_verified_batch_W4_runtime_calls'],
        LM_W4_calls_by_M=lm_runtime['new_LM_W4_calls_by_M'],long_context_graph_evidence=evidence,
        actual_BM8_request_graph_used=any(e['final_KV_source_sha256']==BM8 for e in evidence),
        client_concurrency=64,all42_final_KV_source_bound_to_actual16K_graph=True,
        no_assumption_that_client64_means64_resident_requests=True)
    return result
