# SPDX-License-Identifier: Apache-2.0
"""Offline four-scene SIMT screen audit; no full-boundary or model claim."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from simt_decode import render_source


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve();state=read(root/'result.json')
    assert state['complete'] and state['phase']=='complete' and state['child_exit_code']==0
    assert not state['official_benchmark'] and not state['adopted'] and len(state['frozen_sources'])==10
    assert 'No running processes found' in (root/'hardware_after.txt').read_text()
    repo=Path(__file__).resolve().parents[2]
    for source,digest in state['frozen_sources'].items():
        assert sha(root/'source_snapshot'/Path(source).relative_to('/'))==digest
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):assert sha(repo/Path(source).relative_to('/workspace/vllm-plugin-FL'))==digest
    micro=root/'micro';data=read(micro/'result.json');snap=micro/'source_snapshot'
    assert data['complete'] and data['numerical_checks_passed'] and len(data['cases'])==4
    assert not data.get('compile_errors') and not data.get('changed_input_failures')
    for filename in ('simt_decode.py','probe_simt.py','zero_initialization.py','experimental_cache.py','launch_router.py','triton_unified_attention.py'):
        paths=[p for p in state['frozen_sources'] if Path(p).name==filename];assert len(paths)==1
        assert sha(snap/filename)==state['frozen_sources'][paths[0]]
    cv=micro/'cache_policy_source/generated_cache_policy.py';generated=micro/'simt_source/generated_simt_decode.py'
    assert sha(cv)=='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'
    assert generated.read_text()==render_source(cv.read_bytes())
    generation=data['provenance']['simt_decode_generation'];assert generation['generated_sha256']==sha(generated)
    assert generation['explicit_FP32_sums_of_BF16_products'] and generation['probabilities_cast_to_original_BF16_before_PV']
    assert generation['direct_single_query_grid'] and generation['reducer_source_unchanged']
    assert generation['floating_reduction_order_changed'] and generation['integer_widths_unchanged']
    expected=[(64,4096),(64,5120),(64,65),(8,33)];rows=[];bitwise=True
    for case,(batch,length) in zip(data['cases'],expected):
        assert case['batch']==batch and max(case['kv_lengths'])==length
        assert case['query_row_tiles']==[16]+[8]*5 and len(case['variants'])==5
        assert len(case['changed_graph_bitwise_equal'])==len(case['changed_graph_fp32_checks'])==6
        for group in ('cpu_reference_checks','changed_graph_cpu_checks','changed_graph_fp32_checks'):
            assert case[group] and all(x['passed'] and x['failed_elements']==0 and x['atol']==.02 and x['rtol']==.01 for x in case[group])
        assert case['original_vs_fp32']['passed'] and case['baseline_tail_preserved']
        config=dict(TILE_SIZE=16,num_warps=1,num_stages=1,KV_CACHE_MODIFIER='.cv')
        assert case['baseline_configuration']==config
        segments=8 if batch==64 else 16;queries=sum(case['query_lengths'])
        controls=[(None,1,case['baseline_launch_metadata'],case['baseline_raw_ms'],case['baseline_ms'])]
        plans=[(False,1)]+[(True,w) for w in (1,2,4,8)]
        for (simt,warps),variant in zip(plans,case['variants']):
            assert variant['configuration']==dict(config,BLOCK_M=8,BLOCK_Q=1,SIMT_DECODE=simt,num_warps=warps)
            assert variant['query_row_tile']==8 and variant['tails_preserved']
            assert all(variant[k]['passed'] and variant[k]['failed_elements']==0 for k in ('candidate_vs_fp32','candidate_vs_native'))
            change=100*(variant['candidate_ms']/case['baseline_ms']-1);assert abs(change-variant['time_change_percent'])<1e-10
            controls.append((simt,warps,variant['launch_metadata'],variant['raw_ms'],variant['candidate_ms']))
            rows.append(dict(batch=batch,kv_max=length,SIMT=simt,warps=warps,baseline_ms=case['baseline_ms'],candidate_ms=variant['candidate_ms'],time_change_percent=change,bitwise_equal=variant['candidate_bitwise_equal']))
            bitwise &= variant['candidate_bitwise_equal']
        bitwise &= all(case['changed_graph_bitwise_equal'])
        for simt,warps,metadata,raw,median in controls:
            bm,bq=(16,2) if simt is None else (8,1)
            assert metadata['num_warps']==warps and metadata['num_stages']==1
            assert metadata['actual_BLOCK_M']==bm and metadata['actual_BLOCK_Q']==bq and metadata['actual_BLOCK_SIZE']==16
            assert metadata['actual_grid']==[batch if simt else queries//bq+batch,2,segments]
            assert metadata['producer_and_reducer_tile']==16 and metadata['requested_cache_modifier']=='.cv'
            assert metadata['requested_simt_decode'] is simt and metadata['simt_decode_effective']==bool(simt)
            assert len(raw)==9 and all(t>0 for t in raw) and statistics.median(raw)==median
    artifacts=list((micro/'kernel_artifacts').glob('*/manifest.json'))
    assert {p.parent.name for p in artifacts}=={'baseline_cv','dot_bm8','simt_w1','simt_w2','simt_w4','simt_w8'}
    for path in artifacts:
        record=read(path);assert record['files']
        for name,digest in record['files'].items():assert sha(path.parent/name)==digest
    result=dict(complete=True,official_benchmark=False,adopted=False,core_scenarios=4,candidate_cases=20,recomputed_events=216,
                full_boundary_coverage=False,all10_source_snapshots_verified=True,generated_SIMT_source_verified=True,
                actual_BM8_direct_grid_and_warps_verified=True,original_and_changed_graphs_FP32_CPU_and_tails_verified=True,
                all_original_and_changed_bitwise_equal=bitwise,rows=rows)
    (root/'local_download_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
