# SPDX-License-Identifier: Apache-2.0
"""Native Prefill on 4K/1K, compared to the completed frozen original control."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    numerical = Path('/workspace/benchmark-results/20261012_native_prefill_tianshu/zero_init_full/result.json')
    while not (numerical.exists() and json.loads(numerical.read_text()).get('complete')):
        assert (deadline - datetime.now(timezone.utc)).total_seconds() > 2700
        time.sleep(3)
    data = json.loads(numerical.read_text())
    assert len(data['cases']) == 30 and all(x.get('passed') for x in data['cpu_checks'])
    plan = dict(BLOCK_M=256,TILE_SIZE=128,num_warps=16,num_stages=1)
    selected = [x for x in data['cases'] if x['configuration'] == plan]
    assert len(selected) == 5
    assert all(x.get('candidate_native_vs_fp32',{}).get('passed') and x.get('candidate_output_tail_preserved') for x in selected)
    control = Path('/workspace/benchmark-results/20261012_small_gemm_tianshu/service_c64_4k1k/small_0')
    before = json.loads((control / 'result.json').read_text())
    assert before['phase'] == 'complete' and before['small_gemm'] == '0'
    assert before['workload'] == {'input_tokens':4096,'output_tokens':1024,'concurrency':64,'requests_per_round':128,'rounds':3,'excluded_rounds':[1]}
    assert before['hybrid'] == before['adaptive'] == before['tile'] == '1'
    assert before['buckets'] == before['gather'] == '0' and before['max_batched_tokens'] == 2048
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    shutil.copytree(control,root/'control_0')
    out = root/'native_1'
    out.mkdir()
    command = [sys.executable,'-u',str(Path(__file__).with_name('run_service_trial.py')),
        '--output',str(out),'--native-prefill','1','--buckets','0','--gather','0',
        '--tile','1','--hybrid','1','--adaptive','1','--max-batched-tokens','2048',
        '--requests','128','--concurrency','64','--output-tokens','1024','--rounds','3']
    with (out/'runner.log').open('w') as log:
        code = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    after = json.loads((out/'result.json').read_text())
    assert code == 0 and after['phase'] == 'complete'
    assert before['workload'] == after['workload'] and before['installed_sources'] == after['installed_sources']
    assert all(x['generated_kernel_sha256'] == data['zero_initialization']['generated_sha256'] for x in after['native_prefill_routing_records'])
    means = [statistics.mean(x['total_tokens_per_second'] for x in d['short_4k_rounds'][1:]) for d in (before,after)]
    ttfts = [statistics.mean(x['mean_ttft_seconds']*1000 for x in d['short_4k_rounds'][1:]) for d in (before,after)]
    comparison = {'official_benchmark':False,'independent_quality_pending':True,
        'baseline_tok_s':means[0],'candidate_tok_s':means[1],'throughput_gain_percent':100*(means[1]/means[0]-1),
        'baseline_ttft_ms':ttfts[0],'candidate_ttft_ms':ttfts[1],'ttft_change_percent':100*(ttfts[1]/ttfts[0]-1),
        'workload':before['workload'],'requests_successful':782,'shared_control_source':str(control),
        'frozen_control_sources_and_workload_verified':True,
        'greedy_probes_matched':all(a['text']==b['text'] for a,b in zip(before['rare_cases'],after['rare_cases'])),
        'greedy_probe_count':7,'native_prefill_routing_records':after['native_prefill_routing_records']}
    assert sum(len(d['rare_cases'])+sum(x['successful_requests'] for x in d['short_4k_rounds']) for d in (before,after)) == 782
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison,indent=2))
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'archive_completed_pair.py'),
        '--source',str(root),'--label','native_prefill_standard_20261012'],check=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
