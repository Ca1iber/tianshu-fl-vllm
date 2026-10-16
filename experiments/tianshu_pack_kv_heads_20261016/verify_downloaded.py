# SPDX-License-Identifier: Apache-2.0
"""Recompute packed-head four-scene numerical/timing and source evidence offline."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from pack_kv_heads import render_source


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve();state=read(root/'result.json')
    assert state['phase']=='complete' and state['complete'] and state['child_exit_code']==0
    assert not state['official_benchmark'] and not state['adopted']
    assert len(state['frozen_sources'])==10 and state['requested_candidate_cases']==16
    assert 'No running processes found' in (root/'hardware_after.txt').read_text()
    repo=Path(__file__).resolve().parents[2]
    for source,digest in state['frozen_sources'].items():
        assert sha(root/'source_snapshot'/Path(source).relative_to('/'))==digest
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
            assert sha(repo/Path(source).relative_to('/workspace/vllm-plugin-FL'))==digest
    helper='/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_attention_helpers.py'
    assert state['frozen_sources'][helper]=='59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1'
    micro=root/'micro';data=read(micro/'result.json');assert data['complete'] and data['numerical_checks_passed'] and len(data['cases'])==4
    for filename in ('pack_kv_heads.py','probe_pack.py','zero_initialization.py','experimental_cache.py','launch_router.py','triton_unified_attention.py'):
        matches=[p for p in state['frozen_sources'] if Path(p).name==filename];assert len(matches)==1
        assert sha(micro/'source_snapshot'/filename)==state['frozen_sources'][matches[0]]
    cv=micro/'cache_policy_source/generated_cache_policy.py';gen=micro/'packed_source/generated_pack_kv_heads.py'
    assert sha(cv)=='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'
    assert gen.read_text()==render_source(cv.read_bytes())
    provenance=data['provenance']['packed_heads_generation'];assert provenance['generated_sha256']==sha(gen)
    assert provenance['two_KV_heads_in_one_matrix_tile'] and provenance['cross_head_scores_masked_before_softmax']
    assert provenance['same_logical_tokens_per_iteration'] and provenance['BF16_dot_and_FP32_accumulators_preserved']
    assert provenance['integer_widths_unchanged'] and provenance['epilogue_and_reducer_source_unchanged']
    assert provenance['native_sequence_grid_resolution_preserved']
    rows=[];events=0;cases=[(64,4096),(64,5120),(64,65),(8,33)];successful=0;all_bitwise=True
    for case,(batch,kvlen) in zip(data['cases'],cases):
        assert case['batch']==batch and max(case['kv_lengths'])==kvlen
        assert case['baseline_configuration']==dict(TILE_SIZE=16,num_warps=1,num_stages=1,KV_CACHE_MODIFIER='.cv')
        assert case['original_vs_fp32']['passed'] and case['baseline_tail_preserved']
        assert case['query_row_tiles']==[16]*(1+len(case['variants']))
        for group in ('cpu_reference_checks','changed_graph_cpu_checks','changed_graph_fp32_checks'):
            assert case[group] and all(v['passed'] and v['failed_elements']==0 and v['atol']==.02 and v['rtol']==.01 for v in case[group])
        controls=[(None,1,case['baseline_launch_metadata'],case['baseline_raw_ms'],case['baseline_ms'])]
        assert len(case['changed_graph_bitwise_equal'])==1+len(case['variants'])
        all_bitwise &= all(case['changed_graph_bitwise_equal'])
        identity=[]
        for v in case['variants']:
            config=v['configuration'];packed=config['PACK_KV_HEADS'];warps=config['num_warps']
            assert warps in ((1,2,4) if packed else (1,)) and v['query_row_tile']==16
            assert v['tails_preserved'] and all(v[k]['passed'] and v[k]['failed_elements']==0 for k in ('candidate_vs_fp32','candidate_vs_native'))
            assert config==dict(case['baseline_configuration'],PACK_KV_HEADS=packed,num_warps=warps)
            change=100*(v['candidate_ms']/case['baseline_ms']-1);assert abs(change-v['time_change_percent'])<1e-10
            controls.append((packed,warps,v['launch_metadata'],v['raw_ms'],v['candidate_ms']))
            rows.append(dict(batch=batch,kv_max=kvlen,packed=packed,warps=warps,baseline_ms=case['baseline_ms'],candidate_ms=v['candidate_ms'],time_change_percent=change,bitwise_equal=v['candidate_bitwise_equal']))
            if not packed:identity.append(v)
            successful+=1;all_bitwise &= v['candidate_bitwise_equal']
        assert len(identity)==1
        for packed,warps,metadata,raw,median in controls:
            assert metadata['num_warps']==warps and metadata['num_stages']==1
            assert metadata['actual_BLOCK_M']==16 and metadata['actual_BLOCK_Q']==2 and metadata['actual_BLOCK_SIZE']==16
            assert metadata['actual_grid']==[sum(case['query_lengths'])//2+batch,1 if packed else 2,case['segments']]
            assert metadata['producer_and_reducer_tile']==16 and metadata['logical_KV_tile']==16
            assert metadata['actual_matrix_KV_columns']==(32 if packed else 16)
            assert metadata['requested_pack_kv_heads'] is packed and metadata['requested_cache_modifier']=='.cv'
            assert len(raw)==9 and all(t>0 for t in raw) and statistics.median(raw)==median;events+=9
    assert state['candidate_cases']==successful
    artifacts=list((micro/'kernel_artifacts').glob('*/manifest.json'));assert {p.parent.name for p in artifacts}>={'baseline_cv','identity'}
    for p in artifacts:
        for name,digest in read(p)['files'].items():assert sha(p.parent/name)==digest
    result=dict(complete=True,official_benchmark=False,adopted=False,core_scenarios=4,requested_candidate_cases=16,
        successful_candidate_cases=successful,full_boundary_coverage=False,all10_sources_verified=True,
        actual_logical16_matrix32_and_head_axis1_verified=True,unchanged_reducer_and_integer_widths_verified=True,
        all_original_changed_graphs_FP32_CPU_and_tails_verified=True,raw_events_recomputed=events,
        all_bitwise_equal=all_bitwise,compile_errors=data.get('compile_errors',[]),rows=rows)
    (root/'local_download_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
