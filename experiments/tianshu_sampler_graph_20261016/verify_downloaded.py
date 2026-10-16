# SPDX-License-Identifier: Apache-2.0
"""Recompute full token equality, state flags, guards, sources and micro medians."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--v1-root',type=Path,required=True)
    parser.add_argument('--v2-root',type=Path,required=True);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2];all_results=[]
    for version,root in ((1,args.v1_root.resolve()),(2,args.v2_root.resolve())):
        data=json.loads((root/'result.json').read_text())
        assert not data['official_benchmark'] and not data['adopted']
        assert len(data['source_snapshots'])==(63 if version==1 else 64)
        for source,sha in data['source_snapshots'].items():
            assert hashlib.sha256((root/'source_snapshot'/Path(source).relative_to('/')).read_bytes()).hexdigest()==sha
            if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
                assert hashlib.sha256((repo/Path(source).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
        generated=list((root/'profile/topp_launch_records').rglob('generated_geo.py'))
        assert generated and all(hashlib.sha256(p.read_bytes()).hexdigest()==
            '09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d' for p in generated)
        for index,row in enumerate(data['cases']):
            assert row['index']==index and len(row['expected'])==len(row['actual'])==64
            assert row['tokens_equal']==(row['expected']==row['actual'])
            assert row['rng_equal']==(row['expected_rng_sha256']==row['actual_rng_sha256'])
            assert row['input_unchanged'] and row['seed']==111+index
            assert all(0<=token<130560 for token in row['actual']+row['expected'])
        assert data['state_checks'][0]==dict(name='warmup_capture_preserves_default_rng',passed=True)
        if version==1:
            assert data['phase']=='failed' and not data['complete'] and len(data['cases'])==1
            assert not data['cases'][0]['tokens_equal'] and not data['timings']
            assert data['error'].startswith('AssertionError')
            out=dict(version=1,failed_status_preserved=True,first_tokens_differ=True,all63_sources_verified=True)
        else:
            assert data['phase']=='complete' and data['complete'] and len(data['cases'])==9
            assert all(row['tokens_equal'] and row['rng_equal'] for row in data['cases'])
            mixed=data['state_checks'][1]
            assert len(mixed['outputs'])==len(mixed['expected'])==8
            assert mixed['outputs']==mixed['expected'] and mixed['tokens_equal']
            assert mixed['rng_equal'] and mixed['independent_outputs']
            assert len(data['guard_checks'])==12 and all(v['rejected'] for v in data['guard_checks'])
            assert len(data['timings'])==14
            medians={mode:{metric:statistics.median(v[metric] for v in data['timings'] if v['mode']==mode)
                for metric in ('host_wall_per_call_ms','gpu_stream_per_call_ms')} for mode in ('eager','graph')}
            for mode in ('eager','graph'):
                selected=[v for v in data['timings'] if v['mode']==mode]
                assert len(selected)==7 and {v['repeat'] for v in selected}==set(range(7))
                assert all(v['iterations']==20 and 0<v['gpu_stream_per_call_ms']<=v['host_wall_per_call_ms']+.1 for v in selected)
            assert medians==data['timing_medians']
            out=dict(version=2,all64_sources_verified=True,all576_case_tokens_and512_mixed_tokens_verified=True,
                all12_guard_records_verified=True,timing_medians=medians,
                total_call_time_change_percent=100*(medians['graph']['host_wall_per_call_ms']/medians['eager']['host_wall_per_call_ms']-1),
                all14_timing_records_recomputed=True,service_not_run=True,adopted=False)
        (root/'local_download_audit.json').write_text(json.dumps(out,indent=2)+'\n');all_results.append(out)
    print(json.dumps(all_results))


if __name__=='__main__':main()
