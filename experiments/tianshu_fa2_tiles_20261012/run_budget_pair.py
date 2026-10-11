# SPDX-License-Identifier: Apache-2.0
"""Compare2048/4096 with the same quality-verified tile and native Decode."""
import argparse
import json
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
    quality=json.loads(Path('/workspace/benchmark-results/20261012_fa2_tiles_tianshu/math_tile128/accuracy_audit.json').read_text())
    assert quality['complete'] and quality['accuracy']>.95 and quality['total']==105
    results=[]
    for budget in (2048,4096):
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
        out=root/f'budget_{budget}';out.mkdir(exist_ok=True)
        assert not (out/'state.json').exists()
        command=[sys.executable,'-u',str(Path(__file__).with_name('run_service_trial.py')),
                 '--output',str(out),'--buckets','0','--gather','0','--tile','1',
                 '--hybrid','1','--adaptive','1','--max-batched-tokens',str(budget),
                 '--requests',str(args.requests),'--concurrency','64',
                 '--output-tokens',str(args.output_tokens),'--rounds',str(args.rounds)]
        with (out/'runner.log').open('w') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        data=json.loads((out/'result.json').read_text())
        print(json.dumps({'budget':budget,'phase':data['phase'],'returncode':proc.returncode}),flush=True)
        assert proc.returncode==0 and data['phase']=='complete'
        assert data['tile_hook']['sha256']==quality['tile_hook']['sha256']
        assert all(r['configuration']==quality['tile_configuration'] for r in data['tile_routing_records'])
        for filename,key in (('worker.py','worker_sha256'),('iluvatar_hybrid_attention.py','hybrid_sha256'),
                             ('iluvatar_adaptive_decode.py','adaptive_sha256')):
            assert data['installed_sources'][filename]['sha256']==quality[key]
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
        results.append(data)
    assert results[0]['workload']==results[1]['workload']
    assert results[0]['installed_sources']==results[1]['installed_sources']
    means=[statistics.mean(x['total_tokens_per_second'] for x in r['short_4k_rounds'][1:]) for r in results]
    ttfts=[statistics.mean(x['mean_ttft_seconds']*1000 for x in r['short_4k_rounds'][1:]) for r in results]
    greedy=[{'input_tokens':a['input_tokens'],'matched':a['text']==b['text']}
            for a,b in zip(results[0]['rare_cases'],results[1]['rare_cases'])]
    assert len(greedy)==7
    comparison={'official_benchmark':False,'candidate_quality_pending':True,
                'candidate_budget':4096,'control_budget':2048,'baseline_tok_s':means[0],
                'candidate_tok_s':means[1],'gain_percent':100*(means[1]/means[0]-1),
                'baseline_ttft_ms':ttfts[0],'candidate_ttft_ms':ttfts[1],
                'ttft_change_percent':100*(ttfts[1]/ttfts[0]-1),'greedy_probes':greedy,
                'greedy_probes_matched':all(x['matched'] for x in greedy),
                'variants':[{'budget':r['max_batched_tokens'],'stability':r['measured_round_stability'],
                             'requests_successful':len(r['rare_cases'])+sum(x['successful_requests'] for x in r['short_4k_rounds'])} for r in results]}
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    main()
