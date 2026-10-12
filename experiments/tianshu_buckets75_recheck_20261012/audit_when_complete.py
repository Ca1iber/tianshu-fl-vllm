# SPDX-License-Identifier: Apache-2.0
"""Compare a five-round bucket recheck to the same-profile three-round control."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    candidate_path = root / 'candidate/result.json'
    while not candidate_path.exists():
        assert datetime.now(timezone.utc) < deadline, 'Lease ended before a complete candidate'
        time.sleep(10)
    candidate = json.loads(candidate_path.read_text())
    control = json.loads((args.control / 'result.json').read_text())
    assert candidate['phase'] == control['phase'] == 'complete'
    assert candidate['buckets'] == '1' and control['buckets'] == '0'
    for s in [candidate, control]:
        assert s['gather'] == '0' and s['native_prefill'] == s['adaptive'] == s['decode_launch'] == '1'
        assert s['max_batched_tokens'] == 2048 and s.get('stream_interval', 1) == 1
        assert s['workload']['input_tokens'] == 4096 and s['workload']['output_tokens'] == 1024
        assert s['workload']['concurrency'] == 64 and s['workload']['requests_per_round'] == 128
    assert candidate['decode_hook'] == control['decode_hook'] and candidate['tile_hook'] == control['tile_hook']
    for name in control['installed_sources']:
        assert candidate['installed_sources'][name] == control['installed_sources'][name]
    before, after = control['short_4k_rounds'], candidate['short_4k_rounds']
    assert len(before) == 3 and len(after) == 5
    for rounds in [before, after]:
        for row in rounds:
            assert row['successful_requests'] == len(row['responses']) == 128
            assert all(r['usage']['prompt_tokens'] == 4096 and r['usage']['completion_tokens'] == 1024
                       for r in row['responses'])
    for a, b in zip(before, after):
        assert [r['seed'] for r in a['responses']] == [r['seed'] for r in b['responses']]
    rare_matches = []
    for a, b in zip(control['rare_cases'], candidate['rare_cases']):
        assert (a['seed'], a['input_tokens'], a['output_tokens']) == (b['seed'], b['input_tokens'], b['output_tokens'])
        rare_matches.append(a['text'] == b['text'])
    # Compare the first three rounds on equal workload/seed counts. Keep the
    # extra two candidate rounds exclusively as additional stability evidence.
    means = [statistics.mean(r['total_tokens_per_second'] for r in rows[1:3])
             for rows in [before, after]]
    ttft = [1000 * statistics.mean(r['mean_ttft_seconds'] for r in rows[1:3])
            for rows in [before, after]]
    later = [r['total_tokens_per_second'] for r in after[1:]]
    traces = root / 'candidate/bucket_trace_records'
    configured = [json.loads(p.read_text()) for p in traces.glob('configured_*.json')]
    assert any(s['enabled'] and len(s['points']) == 75 and s['max_capture'] == 2048 for s in configured)
    warmed = [json.loads(p.read_text()) for p in traces.glob('warmup_completed_*.json')]
    assert any(s['complete'] and len(s['calls']) == 75 for s in warmed)
    pad_lines = [json.loads(p.read_text()) for p in traces.glob('padding_*.json')]
    assert pad_lines, 'No actual model row padding observed'
    summary = dict(complete=True, official_benchmark=False, adoption_allowed=False,
                   reference_commit='e7aba7b38f8d4fbfa9f3199121d8ad90c53f98ea',
                   actual_prefill_graph_replay=False,
                   control_reused=True, control_source=str(args.control),
                   paired_rounds=3, paired_excluded_rounds=[1],
                   control_total_tok_s=means[0], candidate_total_tok_s=means[1],
                   paired_gain_percent=100 * (means[1] / means[0] - 1),
                   control_ttft_ms=ttft[0], candidate_ttft_ms=ttft[1],
                   candidate_later_four_mean=statistics.mean(later),
                   candidate_later_four_spread_percent=100 * (max(later) - min(later)) / statistics.mean(later),
                   candidate_tuning_by_round=[len(r['tuning_records']) for r in after],
                   candidate_runtime_tuning_records=candidate['runtime_tune_records'],
                   candidate_startup_tuning_count=len(candidate['startup_tune_records']),
                   candidate_startup_seconds=candidate['startup_seconds'],
                   rare_greedy_matches=rare_matches, startup_bucket_count=75,
                   actual_padding_log=pad_lines,
                   independent_math_for_buckets_not_run=True, original_official_for_buckets_not_run=True)
    (root / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    (root / 'result.json').write_text(json.dumps(dict(phase='complete', official_benchmark=False,
                                                     adoption_allowed=False), indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}), flush=True)
    subprocess.run(['python3', '/workspace/vllm-plugin-FL/experiments/archive_completed_pair.py',
                    '--source', str(root), '--single-probe', '--label', 'decode_stage1_buckets75_20261012'], check=True)


if __name__ == '__main__':
    main()
