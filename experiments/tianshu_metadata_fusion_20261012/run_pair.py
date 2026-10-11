# SPDX-License-Identifier: Apache-2.0
"""Numerically verified integer metadata service control/candidate pair."""
import argparse
import json
import hashlib
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--requests',type=int,default=192)
    parser.add_argument('--output-tokens',type=int,default=128)
    parser.add_argument('--rounds',type=int,default=3)
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    assert not (root/'comparison.json').exists()
    source=Path(__file__).resolve().parent.parent
    audit=Path('/workspace/benchmark-results/20261012_fa2_tiles_tianshu/math_tile128/accuracy_audit.json')
    quality=json.loads(audit.read_text())
    assert quality['complete'] and quality['accuracy']>.95 and quality['total']==105
    assert hashlib.sha256((source/'tianshu_fa2_tiles_20261012/hooks/sitecustomize.py').read_bytes()).hexdigest()==quality['tile_hook']['sha256']
    numeric=json.loads(Path('/workspace/benchmark-results/20261012_metadata_fusion_tianshu/integer_probe/result.json').read_text())
    assert numeric['complete'] and numeric['cases_passed']==numeric['cases_total']
    assert hashlib.sha256(Path(__file__).with_name('experimental_metadata.py').read_bytes()).hexdigest()==numeric['source_sha256']
    results=[]
    for metadata in ('0','1'):
        out=root/f'metadata_{metadata}';out.mkdir(exist_ok=True)
        assert not (out/'state.json').exists(), 'Refusing to overwrite a prior trial'
        command=[sys.executable,'-u',str(Path(__file__).with_name('run_service_trial.py')),
                 '--output',str(out),'--buckets','0','--gather','0','--metadata',metadata,'--tile','1',
                 '--hybrid','1','--adaptive','1','--max-batched-tokens','2048',
                 '--requests',str(args.requests),'--concurrency','64',
                 '--output-tokens',str(args.output_tokens),'--rounds',str(args.rounds)]
        with (out/'runner.log').open('w') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        data=json.loads((out/'result.json').read_text())
        print(json.dumps({'metadata':metadata,'phase':data['phase'],'returncode':proc.returncode}),flush=True)
        assert proc.returncode==0 and data['phase']=='complete'
        assert all(record['configuration']==quality['tile_configuration'] for record in data['tile_routing_records'])
        for filename,key in (('worker.py','worker_sha256'),('iluvatar_hybrid_attention.py','hybrid_sha256'),
                             ('iluvatar_adaptive_decode.py','adaptive_sha256')):
            assert data['installed_sources'][filename]['sha256']==quality[key]
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
        results.append(data)
    assert results[0]['workload']==results[1]['workload']
    assert results[0]['installed_sources']==results[1]['installed_sources']
    assert results[0]['model_runner']==results[1]['model_runner']
    assert results[0]['original_tile_hook']==results[1]['original_tile_hook']
    greedy=[{'input_tokens':before['input_tokens'],'seed':before['seed'],
             'matched':before['text']==after['text']}
            for before,after in zip(results[0]['rare_cases'],results[1]['rare_cases'])]
    assert len(greedy)==len(results[0]['rare_cases'])==len(results[1]['rare_cases'])==7
    means=[];ttfts=[]
    for data in results:
        rows=data['short_4k_rounds'][1:]
        means.append(statistics.mean(r['total_tokens_per_second'] for r in rows))
        ttfts.append(statistics.mean(r['mean_ttft_seconds']*1000 for r in rows))
    comparison={'official_benchmark':False,'independent_quality_pending':True,
                'baseline_tok_s':means[0],'candidate_tok_s':means[1],
                'gain_percent':100*(means[1]/means[0]-1),
                'baseline_ttft_ms':ttfts[0],'candidate_ttft_ms':ttfts[1],
                'ttft_change_percent':100*(ttfts[1]/ttfts[0]-1),
                'greedy_probes':greedy,'greedy_probes_matched':all(x['matched'] for x in greedy),
                'variants':[{'metadata':r['metadata'],'stability':r['measured_round_stability'],
                             'successful_requests':len(r['rare_cases'])+sum(x['successful_requests'] for x in r['short_4k_rounds'])} for r in results]}
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    main()
