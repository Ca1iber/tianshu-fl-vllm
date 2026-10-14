# SPDX-License-Identifier: Apache-2.0
"""Audit source-matched KV16 segment-count numerical and graph timing evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--long-root',type=Path,required=True);p.add_argument('--short-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    base=Path(__file__).resolve().parent;total_variants=cpu=bitwise=0;long_rows=[]
    for kind,root,count in [('long',args.long_root,10),('short',args.short_root,14)]:
        d=json.loads((root/'result.json').read_text());assert d['complete'] and d['numerical_checks_passed']
        assert not d['official_benchmark'] and not d['adoption_allowed']
        assert len(d['cases'])==count and not d.get('compile_errors')
        assert sha(base/'probe_launch.py')==sha(root/'source_snapshot/probe_launch.py')==d['probe_sha256']=='2a75f2b6e73f9895c26985bcfdebb1e363282fdf6b81671d06fc6d7e78658791'
        assert sha(base/'launch_router.py')==sha(root/'source_snapshot/launch_router.py')=='0ebd07262dc93a2bb0d1e6ef13a7c88d30bc5c4ff7eedc5e23a538320dcbd343'
        assert sha(root/'source_snapshot/triton_unified_attention.py')==d['provenance']['native_sha256']=='bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0'
        assert sha(root/'source_snapshot/zero_initialization.py')==sha(base.parent/'tianshu_native_prefill_20261012/zero_initialization.py')=='34f4d388c1c7820a81eb46561858061d351eb7755d62fca804112f08d35aca6e'
        assert sha(root/'zero_initialization/generated_zero_initialization.py')==d['provenance']['zero_initialization']['generated_sha256']=='4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e'
        identities=set()
        for c in d['cases']:
            key=c['batch'],tuple(c['kv_lengths']),c['strided'],tuple(c['query_lengths']);assert key not in identities;identities.add(key)
            assert c['original_vs_fp32']['passed'] and c['baseline_tail_preserved'] and all(x['passed'] for x in c['cpu_reference_checks'])
            cpu+=len(c['cpu_reference_checks']);tile=32 if c['batch']==24 else 16;warps=2 if tile==32 else 1
            segments=16 if c['batch']==8 else 4
            assert c['baseline_tile']==tile and c['baseline_warps']==warps and c['baseline_segments']==segments
            assert c['baseline_launch_metadata']['producer_and_reducer_tile']==tile and c['baseline_launch_metadata']['num_warps']==warps
            assert c['baseline_launch_metadata']['launch_grid'][-1]==segments
            assert len(c['baseline_raw_ms'])==9 and all(x>0 for x in c['baseline_raw_ms'])
            assert abs(statistics.median(c['baseline_raw_ms'])-c['baseline_ms'])<1e-9
            assert len(c['variants'])==3 and {v['configuration']['segment_multiplier'] for v in c['variants']}=={.5,2,4}
            for v in c['variants']:
                assert v['candidate_vs_fp32']['passed'] and v['candidate_vs_native']['passed'] and v['tails_preserved']
                assert v['actual_segments']==int(segments*v['configuration']['segment_multiplier'])
                meta=v['launch_metadata'];assert meta['producer_and_reducer_tile']==tile and meta['num_warps']==warps and meta['num_stages']==1
                assert meta['launch_grid'][:-1]==c['baseline_launch_metadata']['launch_grid'][:-1] and meta['launch_grid'][-1]==v['actual_segments']
                assert len(v['raw_ms'])==9 and all(x>0 for x in v['raw_ms'])
                assert abs(statistics.median(v['raw_ms'])-v['candidate_ms'])<1e-9
                assert abs(100*(v['candidate_ms']/c['baseline_ms']-1)-v['time_change_percent'])<1e-8
                total_variants+=1;bitwise+=v['candidate_bitwise_equal']
            if kind=='long':long_rows.append(dict(batch=c['batch'],kv_max=max(c['kv_lengths']),baseline_ms=c['baseline_ms'],changes=[dict(multiplier=v['configuration']['segment_multiplier'],time_change_percent=v['time_change_percent']) for v in c['variants']]))
    audit=dict(complete=True,scenarios=24,candidate_variants=total_variants,cpu_reference_checks=cpu,bitwise_equal_variants=bitwise,all_numerical_and_tail_checks_passed=True,source_snapshots_verified=True,actual_KV16_guarded_baseline_verified=True,grid_segment_axis_and_matching_reducer_verified=True,graph_replay_timing_scope='9 alternating timings,50 replays each',official_benchmark=False,adopted=False,long_rows=long_rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
