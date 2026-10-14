# SPDX-License-Identifier: Apache-2.0
"""Audit register limits, actual driver metadata, numerical checks and raw events."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    data=json.loads((root/'result.json').read_text())
    assert data['complete'] and len(data['cases'])==17
    assert not data['official_benchmark'] and not data['adoption_allowed']
    assert data['b64_segments']==8 and not data['provenance']['grid_changed']
    assert not data['provenance']['installed_sources_modified']
    base=Path(__file__).resolve().parent;snap=root/'source_snapshot'
    assert hashlib.sha256((base/'probe_registers.py').read_bytes()).hexdigest()==data['probe_sha256']
    assert hashlib.sha256((snap/'probe_registers.py').read_bytes()).hexdigest()==data['probe_sha256']
    assert hashlib.sha256((snap/'triton_unified_attention.py').read_bytes()).hexdigest()=='bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0'
    for rel in ('tianshu_decode_low_warps_20261014/launch_router.py','tianshu_native_prefill_20261012/zero_initialization.py'):
        source=base.parent/rel
        assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256((snap/source.name).read_bytes()).digest()
    assert any(hashlib.sha256(p.read_bytes()).hexdigest()=='4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e' for p in (root/'zero_initialization').glob('*.py'))
    assert len(data['backend_source_sha256'])==2
    for name,digest in data['backend_source_sha256'].items():
        assert hashlib.sha256((snap/name).read_bytes()).hexdigest()==digest
    assert 'iluvatar-num-vgpr' in (snap/'backend_compiler.py').read_text()
    assert 'self.n_regs, self.n_spills' in (snap/'compiled_kernel.py').read_text()
    expected=[(b,4096) for b in (8,16,24,32,64)]+[(64,5120),(64,65)]+[(8,n) for n in (1,15,16,17,31,32,33,63,64,65)]
    rows=[];valid={64:True,96:True,128:True}
    for item,(batch,length) in zip(data['cases'],expected):
        assert item['batch']==batch and max(item['kv_lengths'])==length
        queries=sum(item['query_lengths']);segments=8 if batch==64 else 4 if batch>=16 else 16
        tile=16 if batch in (8,16,32,40,48,56,64) else 32;warps=1 if tile==16 else 2
        assert item['baseline_configuration']==dict(TILE_SIZE=tile,num_warps=warps,num_stages=1)
        assert item['logical_KV_identical'] and item['query_row_tiles']==[16,16,16,16]
        assert item['baseline_tail_preserved'] and item['original_vs_fp32']['passed']
        for group in ('cpu_reference_checks','changed_graph_fp32_checks','changed_graph_cpu_checks'):
            assert item[group] and all(v['passed'] and v['failed_elements']==0 and v['atol']==.02 and v['rtol']==.01 for v in item[group])
        assert len(item['changed_graph_fp32_checks'])==1+len(item['variants'])
        meta=item['baseline_launch_metadata']
        assert meta['num_warps']==warps and meta['num_stages']==1
        assert meta['actual_BLOCK_M']==16 and meta['actual_BLOCK_Q']==2 and meta['actual_BLOCK_SIZE']==16
        assert meta['actual_grid']==[queries//2+batch,2,segments] and meta['producer_and_reducer_tile']==tile
        before=statistics.median(item['baseline_raw_ms'])
        assert len(item['baseline_raw_ms'])==9 and before==item['baseline_ms']
        stages=set()
        for candidate in item['variants']:
            stage=candidate['configuration']['maxnreg']
            assert stage in (64,96,128) and stage not in stages;stages.add(stage)
            assert candidate['configuration']==dict(item['baseline_configuration'],maxnreg=stage)
            assert candidate['candidate_vs_fp32']['passed'] and candidate['candidate_vs_native']['passed'] and candidate['tails_preserved']
            meta=candidate['launch_metadata']
            assert meta['num_warps']==warps and meta['num_stages']==1
            assert meta['requested_maxnreg']==stage and meta['compiled_maxnreg']==stage
            assert meta['actual_BLOCK_M']==16 and meta['actual_BLOCK_Q']==2 and meta['actual_BLOCK_SIZE']==16
            assert meta['actual_grid']==[queries//2+batch,2,segments] and meta['producer_and_reducer_tile']==tile
            raw=candidate['raw_ms'];assert len(raw)==9 and all(x>0 for x in raw)
            after=statistics.median(raw);assert after==candidate['candidate_ms']
            change=100*(after/before-1);assert abs(change-candidate['time_change_percent'])<1e-10
            rows.append(dict(batch=batch,kv_max=length,requested_maxnreg=stage,baseline_n_regs=item['baseline_launch_metadata']['n_regs'],
                             baseline_n_spills=item['baseline_launch_metadata']['n_spills'],candidate_n_regs=meta['n_regs'],
                             candidate_n_spills=meta['n_spills'],baseline_ms=before,candidate_ms=after,
                             time_change_percent=change,bitwise_equal=candidate['candidate_bitwise_equal']))
        for stage in (64,96,128):
            if stage not in stages:valid[stage]=False
    assert data['valid_register_limits']==[s for s in (64,96,128) if valid[s]]
    assert data['numerical_checks_passed']==all(valid.values())
    assert 'No running processes found' in (root/'hardware_after.txt').read_text()
    audit=dict(complete=True,official_benchmark=False,adopted=False,source6_verified=True,scenarios=17,
               candidate_cases=len(rows),bitwise_equal_candidates=sum(r['bitwise_equal'] for r in rows),
               valid_register_limits=data['valid_register_limits'],compile_errors=len(data.get('compile_errors',[])),
               changed_Graph_GPU_FP32_and_CPU_samples_verified=True,actual_compiled_grid_and_limit_verified=True,
               all_tails_verified=True,raw_event_values_recomputed=9*(17+len(rows)),rows=rows,
               scope='Compiler maxnreg and driver metadata only; no occupancy/service/quality claim')
    (root/'local_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='rows'}))


if __name__=='__main__':main()
