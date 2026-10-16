# SPDX-License-Identifier: Apache-2.0
"""Audit the fresh public/Torch indexing checks from downloaded artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root.resolve()
    d=json.loads((root/'result.json').read_text());api=json.loads((root/'dispatch/result.json').read_text())
    assert d['phase']=='complete' and d['complete'] and len(d['source_snapshots'])==4
    repo=Path(__file__).resolve().parents[2]
    for name,h in d['source_snapshots'].items():
        assert hashlib.sha256((root/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==h
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==h
    assert api['complete'] and api['passed'] and len(api['cases'])==48
    expected={(shape,dtype,idx,strided) for shape in ((168,),(193,2048),(129,130560)) for dtype in ('torch.bfloat16','torch.float32','torch.int32','torch.int64') for idx in ('torch.int32','torch.int64') for strided in (False,True)}
    assert {(tuple(r['shape']),r['dtype'],r['index_dtype'],r['strided']) for r in api['cases']}==expected
    assert all(r['actual_api_fixed_route'] and r['bitwise_cpu_equal'] for r in api['cases'])
    assert [r['shift'] for r in api['graph_replays']]==[1,17,46] and all(r['bitwise_cpu_equal'] for r in api['graph_replays'])
    rows=d['torch_api_cases'];assert len(rows)==6
    assert {(r['width'],r['count'],r['dtype']) for r in rows}=={(w,n,t) for w,n in ((1,29),(2048,69),(130560,18)) for t in ('torch.int32','torch.bfloat16')}
    assert all(r['actual_Torch_API_fixed_route'] and r['bitwise_CPU_equal'] for r in rows)
    assert len(d['actual_torch_API_calls'])==6
    report=dict(complete=True,all48_public_API_checks=True,all3_changed_Graph_checks=True,all6_Torch_API_routes=True,
        all4_source_snapshots_verified=True,full_raw_GPU_buffers_not_serialized=True,
        remote_result='/workspace/benchmark-results/'+root.name+'/result.json',
        remote_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
        no_model_or_quality_claim=True,adopted=False)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
