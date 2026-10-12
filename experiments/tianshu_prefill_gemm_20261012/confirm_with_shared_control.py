# SPDX-License-Identifier: Apache-2.0
"""Confirm M2048 GEMM on 4K/1K using the completed identical M8-off control."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    first = json.loads(Path('/workspace/benchmark-results/20261012_prefill_gemm_tianshu/refill128/comparison.json').read_text())
    assert first['throughput_gain_percent'] > .5
    baseline = json.loads((args.control / 'result.json').read_text())
    assert baseline['phase'] == 'complete' and baseline['small_gemm'] == '0'
    assert baseline['workload'] == {'input_tokens':4096,'output_tokens':1024,'concurrency':64,'requests_per_round':128,'rounds':3,'excluded_rounds':[1]}
    assert baseline['adaptive'] == baseline['hybrid'] == baseline['tile'] == '1'
    assert baseline['max_batched_tokens'] == 2048 and baseline['buckets'] == baseline['gather'] == '0'
    assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    shutil.copytree(args.control, root / 'control_0')
    candidate_dir = root / 'prefill_1'
    candidate_dir.mkdir()
    command = [sys.executable, '-u', str(Path(__file__).with_name('run_service_trial.py')),
               '--output', str(candidate_dir), '--prefill-gemm','1','--buckets','0','--gather','0',
               '--tile','1','--hybrid','1','--adaptive','1','--max-batched-tokens','2048',
               '--requests','128','--concurrency','64','--output-tokens','1024','--rounds','3']
    with (candidate_dir / 'runner.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    candidate = json.loads((candidate_dir / 'result.json').read_text())
    assert code == 0 and candidate['phase'] == 'complete'
    assert baseline['workload'] == candidate['workload']
    assert baseline['installed_sources'] == candidate['installed_sources']
    means = [statistics.mean(x['total_tokens_per_second'] for x in d['short_4k_rounds'][1:]) for d in (baseline, candidate)]
    ttfts = [statistics.mean(x['mean_ttft_seconds'] * 1000 for x in d['short_4k_rounds'][1:]) for d in (baseline, candidate)]
    result = {'official_benchmark':False,'independent_quality_pending':True,
              'baseline_tok_s':means[0],'candidate_tok_s':means[1],'throughput_gain_percent':100*(means[1]/means[0]-1),
              'baseline_ttft_ms':ttfts[0],'candidate_ttft_ms':ttfts[1],'ttft_change_percent':100*(ttfts[1]/ttfts[0]-1),
              'workload':baseline['workload'],'shared_control_source':str(args.control),
              'control_sources_and_workload_verified_identical':True,
              'greedy_probes_matched':all(a['text']==b['text'] for a,b in zip(baseline['rare_cases'],candidate['rare_cases'])),
              'greedy_probe_count':7,'requests_successful':782,
              'actual_prefill_mm_records':candidate['prefill_mm_routing_records']}
    assert sum(len(d['rare_cases']) + sum(x['successful_requests'] for x in d['short_4k_rounds']) for d in (baseline,candidate)) == 782
    (root / 'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'archive_completed_pair.py'),
                    '--source',str(root),'--label','prefill_gemm_c64_4k1k_20261012'],check=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
