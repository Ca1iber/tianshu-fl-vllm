# SPDX-License-Identifier: Apache-2.0
"""Summarize serialized token-budget trials without claiming official scores."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def summarize(root, budgets=(2048, 4096, 8192), expected_rounds=5):
    variants = {}
    greedy = {}
    workload = None
    for budget in budgets:
        directory = root / f'budget_{budget}'
        result = json.loads((directory / 'result.json').read_text())
        assert result['phase'] == 'complete', result['phase']
        assert 'No running processes found' in (directory / 'hardware_after.txt').read_text()
        rounds = result['short_4k_rounds']
        assert len(rounds) == result['workload']['rounds'] == expected_rounds
        if workload is None:
            workload = result['workload']
        assert result['workload'] == workload
        assert all(r['successful_requests'] == result['workload']['requests_per_round'] for r in rounds)
        measured = rounds[1:]
        responses = [response for row in measured for response in row['responses']]
        log = (directory / 'server.log').read_text(errors='replace')
        capacity = re.findall(r'GPU KV cache size: ([\d,]+) tokens', log)
        assert len(capacity) == 1, capacity
        greedy[budget] = {row['input_tokens']: hashlib.sha256(row['text'].encode()).hexdigest()
                          for row in result['rare_cases']}
        variants[budget] = {
            'mean_total_tok_s': statistics.mean(r['total_tokens_per_second'] for r in measured),
            'mean_output_tok_s': statistics.mean(r['output_tokens_per_second'] for r in measured),
            'mean_ttft_ms': 1000 * statistics.mean(r['ttft_seconds'] for r in responses),
            'mean_tpot_ms': 1000 * statistics.mean(
                (r['wall_seconds'] - r['ttft_seconds']) / (r['output_tokens'] - 1) for r in responses),
            'mean_request_wall_seconds': statistics.mean(r['wall_seconds'] for r in responses),
            'kv_cache_tokens': int(capacity[0].replace(',', '')),
            'startup_seconds': result['startup_seconds'],
            'stability': result['measured_round_stability'],
            'successful_requests': sum(r['successful_requests'] for r in rounds) + len(result['rare_cases']),
            'adaptive': result['adaptive'],
            'installed_sources': result['installed_sources'],
        }
    baseline = variants[2048]
    for budget, data in variants.items():
        data['total_tok_s_gain_percent'] = 100 * (data['mean_total_tok_s'] / baseline['mean_total_tok_s'] - 1)
        data['ttft_change_percent'] = 100 * (data['mean_ttft_ms'] / baseline['mean_ttft_ms'] - 1)
        data['tpot_change_percent'] = 100 * (data['mean_tpot_ms'] / baseline['mean_tpot_ms'] - 1)
        data['greedy_probe_matches'] = sum(value == greedy[2048][key] for key, value in greedy[budget].items())
        data['greedy_probe_count'] = len(greedy[budget])
        assert data['adaptive'] == baseline['adaptive']
        assert data['installed_sources'] == baseline['installed_sources']
    return {
        'official_benchmark': False,
        'workload': workload,
        'excluded_rounds': [1],
        'quality_validation_required_before_adoption': True,
        'tpot_definition': '(client request wall time - first nonempty text time)/(output tokens - 1)',
        'variants': variants,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', choices=[2048, 4096, 8192],
                        default=[2048, 4096, 8192])
    parser.add_argument('--expected-rounds', type=int, default=5)
    args = parser.parse_args()
    if len(set(args.budgets)) != len(args.budgets) or args.budgets[0] != 2048:
        parser.error('Use unique budgets beginning with the2048 control')
    if len(args.budgets) < 2 or args.expected_rounds < 2:
        parser.error('At least two variants and two rounds are required')
    root = args.output.resolve()
    summary = summarize(root, args.budgets, args.expected_rounds)
    (root / 'budget_comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
