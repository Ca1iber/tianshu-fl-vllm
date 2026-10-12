# SPDX-License-Identifier: Apache-2.0
"""One-factor budget confirmation with a frozen, native-prefill N128 control."""
import argparse
import hashlib
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
    parser.add_argument('--budget', type=int, choices=[4096, 8192], required=True)
    args = parser.parse_args()
    baseline = json.loads((args.control/'result.json').read_text())
    assert baseline['phase']=='complete' and baseline['native_prefill']=='1'
    assert baseline['adaptive']==baseline['hybrid']==baseline['tile']=='1'
    assert baseline['buckets']==baseline['gather']=='0' and baseline['max_batched_tokens']==2048
    assert baseline['workload']==dict(input_tokens=4096, output_tokens=1024,
                                      concurrency=64, requests_per_round=128,
                                      rounds=3, excluded_rounds=[1])
    assert all(not r['tuning_records'] for r in baseline['short_4k_rounds'])
    directory = Path(__file__).resolve().parent
    hook = directory/'hooks/sitecustomize.py'
    assert hashlib.sha256(hook.read_bytes()).hexdigest()==baseline['tile_hook']['sha256']
    assert baseline['native_prefill_routing_records']
    assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    shutil.copytree(args.control, root/'control_2048')
    candidate_dir = root/f'candidate_{args.budget}'
    command = [sys.executable, '-u', str(directory/'run_service_trial.py'),
               '--output', str(candidate_dir), '--native-prefill', '1', '--buckets', '0',
               '--gather', '0', '--tile', '1', '--hybrid', '1', '--adaptive', '1',
               '--max-batched-tokens', str(args.budget), '--concurrency', '64',
               '--requests', '128', '--output-tokens', '1024', '--rounds', '3']
    # The service trial owns creation of its output directory and requires it
    # not to exist. Keep the launcher log in the parent comparison directory.
    with (root/'candidate_runner.log').open('w') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    assert process.returncode==0, f'Candidate exited {process.returncode}; see candidate_runner.log'
    candidate = json.loads((candidate_dir/'result.json').read_text())
    assert process.returncode==0 and candidate['phase']=='complete'
    assert baseline['workload']==candidate['workload']
    assert baseline['installed_sources']==candidate['installed_sources']
    assert baseline['tile_hook']==candidate['tile_hook']
    generated = {r['generated_kernel_sha256'] for r in baseline['native_prefill_routing_records']}
    assert all(r['generated_kernel_sha256'] in generated for r in candidate['native_prefill_routing_records'])
    groups = [baseline, candidate]
    means = [statistics.mean(r['total_tokens_per_second'] for r in d['short_4k_rounds'][1:]) for d in groups]
    ttfts = [statistics.mean(r['mean_ttft_seconds']*1000 for r in d['short_4k_rounds'][1:]) for d in groups]
    success = sum(len(d['rare_cases'])+sum(r['successful_requests'] for r in d['short_4k_rounds']) for d in groups)
    assert success==782
    result = dict(official_benchmark=False, independent_quality_pending=True,
                  adopted_as_best=False, control_budget=2048, candidate_budget=args.budget,
                  shared_control_source=str(args.control), workload=baseline['workload'],
                  baseline_tok_s=means[0], candidate_tok_s=means[1],
                  throughput_gain_percent=100*(means[1]/means[0]-1),
                  baseline_ttft_ms=ttfts[0], candidate_ttft_ms=ttfts[1],
                  ttft_change_percent=100*(ttfts[1]/ttfts[0]-1),
                  requests_successful=success,
                  greedy_probes_matched=all(a['text']==b['text'] for a,b in zip(baseline['rare_cases'], candidate['rare_cases'])),
                  timed_round_tuning_counts=[[len(r['tuning_records']) for r in d['short_4k_rounds']] for d in groups],
                  postfirst_total_tok_s=[[r['total_tokens_per_second'] for r in d['short_4k_rounds'][1:]] for d in groups],
                  candidate_routing_records=candidate['native_prefill_routing_records'])
    (root/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    subprocess.run([sys.executable, str(directory.parent/'archive_completed_pair.py'),
                    '--source', str(root), '--label', f'native_prefill_budget{args.budget}_20261012'], check=True)


if __name__=='__main__':
    main()
