# SPDX-License-Identifier: Apache-2.0
"""C8 continuous serving pair for the three numerically verified M8 shapes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    numeric=Path('/workspace/benchmark-results/20261012_next_diagnostics_tianshu/splitk_v2_full/result.json')
    data=json.loads(numeric.read_text())
    assert data['complete'] and len(data['cases'])==120
    assert all(c.get('vs_fp32',{}).get('passed') and c.get('vs_cpu',{}).get('passed') for c in data['cases'])
    kernel=Path(__file__).resolve().parents[1]/'tianshu_splitk_20261012/experimental_mm.py'
    assert hashlib.sha256(kernel.read_bytes()).hexdigest()==data['source_sha256']
    results=[]
    for flag in ('0','1'):
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
        out=root/f'small_{flag}';out.mkdir(exist_ok=True);assert not (out/'state.json').exists()
        command=[sys.executable,'-u',str(Path(__file__).with_name('run_service_trial.py')),
                 '--output',str(out),'--small-gemm',flag,'--buckets','0','--gather','0',
                 '--tile','1','--hybrid','1','--adaptive','1','--max-batched-tokens','2048',
                 '--requests','16','--concurrency','8','--output-tokens','1024','--rounds','3']
        with (out/'runner.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads((out/'result.json').read_text());print(json.dumps({'small_gemm':flag,'phase':result['phase'],'exit_code':r.returncode}),flush=True)
        assert r.returncode==0 and result['phase']=='complete'
        results.append(result)
    assert results[0]['workload']==results[1]['workload']
    assert results[0]['installed_sources']==results[1]['installed_sources']
    means=[statistics.mean(x['total_tokens_per_second'] for x in r['short_4k_rounds'][1:]) for r in results]
    ttfts=[statistics.mean(x['mean_ttft_seconds']*1000 for x in r['short_4k_rounds'][1:]) for r in results]
    greedy=[a['text']==b['text'] for a,b in zip(results[0]['rare_cases'],results[1]['rare_cases'])]
    result={'official_benchmark':False,'independent_quality_pending':True,'concurrency':8,'requests_per_round':16,
            'baseline_tok_s':means[0],'candidate_tok_s':means[1],'throughput_gain_percent':100*(means[1]/means[0]-1),
            'baseline_ttft_ms':ttfts[0],'candidate_ttft_ms':ttfts[1],'ttft_change_percent':100*(ttfts[1]/ttfts[0]-1),
            'greedy_probes_matched':all(greedy),'greedy_probe_count':len(greedy),
            'requests_successful':sum(len(r['rare_cases'])+sum(x['successful_requests'] for x in r['short_4k_rounds']) for r in results),
            'actual_small_mm_records':results[1]['small_mm_routing_records']}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
