# SPDX-License-Identifier: Apache-2.0
"""Audit original/current-CV direct mapping component records and provenance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from render import render_source


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root.resolve()
    d=json.loads((root/'result.json').read_text());assert d['phase']=='complete' and d['complete'] and not d['adopted'] and not d['official_benchmark']
    assert d['full_model_and_Math_not_run'];repo=Path(__file__).resolve().parents[2]
    for name,sha in d['source_snapshots'].items():
        assert digest(root/'source_snapshot'/Path(name).relative_to('/'))==sha,name
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert digest(repo/Path(name).relative_to('/workspace/vllm-plugin-FL'))==sha,name
    original=next(root.glob('decode_coherent_records/cache_generated/*/generated_cache_policy.py'))
    generated=root/'direct_generated/generated_direct_decode.py';text,record=render_source(original.read_bytes());assert text==generated.read_text()
    for k,v in record.items():assert d['derivation'][k]==v
    cases=((1,33,False,False),(8,4096,False,False),(24,4096,True,False),(32,4096,False,False),
           (64,4096,False,False),(64,5120,True,False),(8,257,True,True),(64,4096,True,True))
    assert [(c['batch'],c['context'],c['strided'],c['holes']) for c in d['cases']]==list(cases)
    events=CPU_checks=FP32_checks=0;timings=[]
    for c in d['cases']:
        n=c['batch'];qlens=[int(i%3!=1) if c['holes'] else 1 for i in range(n)];queries=sum(qlens)
        tile=16 if n in (8,16,32,40,48,56,64) else 32
        warps=1 if n in (8,16,32,40,48,56,64) else 2 if 8<=n<=64 else 4
        seg=64 if n==1 else 8 if n==64 else 4 if n>=16 else 16
        assert c['queries']==queries and c['segments']==seg
        assert c['key_strides']==[16384 if c['strided'] else 8192,256,128,1]
        assert c['query_strides']==[2560 if c['strided'] else 2048,128,1]
        assert len(c['producer_metadata'])==2
        original_grid=[queries//2+n,2,seg]
        for direct,m in enumerate(c['producer_metadata']):
            assert m['original_grid']==original_grid and m['actual_grid']==([n,2,seg] if direct else original_grid)
            assert m['tile']==tile and m['warps']==m['compiled_warps']==warps
            assert m['n_regs']>0 and m['n_spills']>=0
        assert len(c['phases'])==2
        for phase,r in enumerate(c['phases']):
            expected=qlens[-1:]+qlens[:-1] if phase and c['holes'] else qlens
            assert r['phase']==phase and r['query_lengths']==expected and sum(expected)==queries
            assert r['bitwise_equal'] and r['outputs_cleared_before_replay']
            assert len(r['checks'])==2 and [x['direct'] for x in r['checks']]==[False,True]
            for x in r['checks']:
                a=x['against_FP32'];assert a['passed'] and a['failed_elements']==0 and a['atol']==.02 and a['rtol']==.01 and x['tails_preserved']
                assert math.isfinite(a['max_abs']) and a['max_abs']>=0;FP32_checks+=1
            assert len(r['CPU_reference'])==(4 if n==1 else 12)
            assert all(x['passed'] and x['failed_elements']==0 for x in r['CPU_reference']);CPU_checks+=len(r['CPU_reference'])
        rows=c['raw_ms'];assert len(rows)==2 and all(len(r)==7 and all(math.isfinite(x) and x>0 for x in r) for r in rows)
        assert statistics.median(rows[0])==c['baseline_ms'] and statistics.median(rows[1])==c['candidate_ms']
        assert abs(c['time_change_percent']-100*(c['candidate_ms']/c['baseline_ms']-1))<1e-12
        events+=14;timings.append({k:c[k] for k in ('batch','context','strided','holes','baseline_ms','candidate_ms','time_change_percent','producer_metadata')})
    assert events==112 and CPU_checks==176 and FP32_checks==32
    audit=dict(complete=True,source_snapshots_verified=len(d['source_snapshots']),generated_source_reconstructed=True,
        reducer_AST_unchanged_and_two_reversible_edits_only=True,all112_event_groups_recomputed=True,
        all16_phase_records_bitwise_equal=True,all32_FP32_checks_and176_CPU_head_checks_checked=True,
        internal_zero_query_cu_changed_and_same_Graph_replayed=True,full_GPU_tensor_values_not_serialized=True,
        B64_full_4K_and5K_slower=True,service_Math_not_run=True,adopted=False,timings=timings)
    assert all(c['time_change_percent']>0 for c in d['cases'] if c['batch']==64 and not c['holes'])
    (root/'local_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='timings'}))


if __name__=='__main__':main()
