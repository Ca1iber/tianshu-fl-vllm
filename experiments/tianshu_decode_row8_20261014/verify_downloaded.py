# SPDX-License-Identifier: Apache-2.0
"""Audit the negative row8 probe, numerical checks, actual grids and raw events."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    data=json.loads((root/'result.json').read_text())
    assert data['complete'] and data['numerical_checks_passed']
    assert not data['official_benchmark'] and not data['adoption_allowed']
    assert data['valid_row_tiles']==[16,8] and len(data['cases'])==17
    assert not data.get('compile_errors') and data['b64_segments']==8
    base=Path(__file__).resolve().parent
    local=base/'probe_rows.py'
    assert hashlib.sha256(local.read_bytes()).hexdigest()==data['probe_sha256']
    snap=root/'source_snapshot'
    assert hashlib.sha256((snap/'probe_rows.py').read_bytes()).hexdigest()==data['probe_sha256']
    assert hashlib.sha256((snap/'triton_unified_attention.py').read_bytes()).hexdigest()=='bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0'
    for rel in ('tianshu_decode_low_warps_20261014/launch_router.py','tianshu_native_prefill_20261012/zero_initialization.py'):
        actual=base.parent/rel
        assert hashlib.sha256(actual.read_bytes()).digest()==hashlib.sha256((snap/actual.name).read_bytes()).digest()
    provenance=data['provenance']
    assert provenance['grid_changed'] and not provenance['installed_sources_modified']
    assert provenance['producer_and_reducer_tile_matched']
    generated=list((root/'zero_initialization').rglob('*.py'))
    assert any(hashlib.sha256(p.read_bytes()).hexdigest()=='4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e' for p in generated)
    rows=[]
    expected=[(b,4096) for b in (8,16,24,32,64)]+[(64,5120),(64,65)]+[(8,n) for n in (1,15,16,17,31,32,33,63,64,65)]
    for item,(batch,length) in zip(data['cases'],expected):
        assert item['batch']==batch and max(item['kv_lengths'])==length
        queries=sum(item['query_lengths'])
        assert len(item['query_lengths'])==len(item['kv_lengths'])==batch
        assert item['logical_KV_identical'] and item['query_row_tiles']==[16,8]
        assert item['original_vs_fp32']['passed'] and item['baseline_tail_preserved']
        for group in ('cpu_reference_checks','changed_graph_fp32_checks','changed_graph_cpu_checks'):
            assert item[group] and all(v['passed'] and v['failed_elements']==0 and v['atol']==.02 and v['rtol']==.01 for v in item[group])
        assert len(item['changed_graph_fp32_checks'])==2 and len(item['variants'])==1
        candidate=item['variants'][0]
        assert candidate['candidate_vs_fp32']['passed'] and candidate['candidate_vs_native']['passed']
        assert candidate['candidate_bitwise_equal'] and candidate['tails_preserved']
        selected=batch in (8,16,32,40,48,56,64)
        tile=16 if selected else 32
        warps=1 if selected else 2
        assert item['baseline_configuration']==dict(TILE_SIZE=tile,num_warps=warps,num_stages=1)
        assert candidate['configuration']==dict(item['baseline_configuration'],BLOCK_M=8,BLOCK_Q=1)
        segments=8 if batch==64 else 4 if batch>=16 else 16
        assert item['segments']==segments
        for meta,bm,bq in ((item['baseline_launch_metadata'],16,2),(candidate['launch_metadata'],8,1)):
            assert meta['num_warps']==warps and meta['num_stages']==1
            assert meta['actual_BLOCK_M']==bm and meta['actual_BLOCK_Q']==bq and meta['actual_BLOCK_SIZE']==16
            assert meta['actual_grid']==[queries//bq+batch,2,segments]
            assert meta['producer_and_reducer_tile']==tile
        raw=[item['baseline_raw_ms'],candidate['raw_ms']]
        assert all(len(v)==9 and all(x>0 for x in v) for v in raw)
        medians=list(map(statistics.median,raw))
        assert medians==[item['baseline_ms'],candidate['candidate_ms']]
        change=100*(medians[1]/medians[0]-1)
        assert abs(change-candidate['time_change_percent'])<1e-10
        rows.append(dict(batch=batch,kv_max=length,baseline_ms=medians[0],candidate_ms=medians[1],time_change_percent=change))
    assert 'No running processes found' in (root/'hardware_after.txt').read_text()
    audit=dict(complete=True,official_benchmark=False,adopted=False,source4_verified=True,
               cases=17,bitwise_equal_cases=17,changed_Graph_GPU_FP32_and_CPU_samples_verified=True,
               grid_and_compiled_metadata_verified=True,all_tails_verified=True,
               raw_event_values_recomputed=306,rows=rows,scope='Synthetic row8 negative; no model throughput/quality claim')
    (root/'local_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
