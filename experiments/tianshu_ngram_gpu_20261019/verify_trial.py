# SPDX-License-Identifier: Apache-2.0
"""Audit source, all diagnostic responses, latency counters and graph coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
from audit_profile import audit_runtime


def metrics(path):
    found = {}
    for line in path.read_text().splitlines():
        m = re.match(r'^(vllm:[a-zA-Z0-9_]+)(?:\{[^}]*\})? ([0-9.eE+\-]+)$', line)
        if m:
            found[m[1]] = found.get(m[1], 0.) + float(m[2])
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    state = json.loads((root / 'result.json').read_text())
    assert state['phase'] == 'complete' and state['ngram_gpu_service_screen']
    runtime = audit_runtime(root, state, online=False)
    repo = Path(__file__).resolve().parents[2]
    for name, sha in state['frozen_sources'].items():
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert hashlib.sha256((repo / Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest() == sha
    rows = state['short_4k_rounds']
    assert len(rows) == 3
    counters = []
    seeds = set()
    for i, row in enumerate(rows):
        responses = row['responses']
        assert len(responses) == row['successful_requests'] == 64 and row['round'] == i + 1
        expected = [202610192000 + 100 * i + j for j in range(64)]
        assert [r['input_seed'] for r in responses] == expected
        assert not seeds.intersection(expected)
        seeds.update(expected)
        for r in responses:
            assert r['api_seed'] is None and r['input_tokens'] == 4096 and r['output_tokens'] == 1024
            assert r['usage']['prompt_tokens'] == 4096 and r['usage']['completion_tokens'] == 1024
            assert r['usage']['total_tokens'] == 5120 and r['text']
            assert 0 < r['ttft_seconds'] <= r['wall_seconds'] <= row['duration_seconds'] + .1
        assert row['total_tokens_per_second'] == 64 * 5120 / row['duration_seconds']
        assert row['output_tokens_per_second'] == 64 * 1024 / row['duration_seconds']
        assert row['mean_ttft_seconds'] == statistics.mean(r['ttft_seconds'] for r in responses)
        before = metrics(root / f'round{i+1}_before_metrics.txt')
        after = metrics(root / f'round{i+1}_after_metrics.txt')
        delta = {k: after[k] - v for k, v in before.items() if k in after}
        assert delta['vllm:request_generation_tokens_count'] == 64
        assert delta['vllm:request_generation_tokens_sum'] == 64 * 1024
        assert delta['vllm:request_prompt_tokens_sum'] == 64 * 4096
        assert delta['vllm:time_to_first_token_seconds_count'] == 64
        counters.append({k: v for k, v in delta.items() if k.startswith('vllm:spec_decode_')})
    canaries = state['seeded_canaries'] + state['flush_canaries']
    assert len(state['seeded_canaries']) == 2 and len(state['flush_canaries']) == 3
    assert len({c['text'] for c in canaries}) == 1
    for c in canaries:
        assert c['api_seed'] == 777 and c['input_seed'] == 2026101699
        assert c['usage']['prompt_tokens'] == 128 and c['usage']['completion_tokens'] == 16
    assert state['mean_total_tok_s'] == statistics.mean(r['total_tokens_per_second'] for r in rows[1:])
    report = dict(complete=True, all192_screen_and5_canary_responses_checked=True,
                  all_sources_and_actual42_environment_checked=True, runtime=runtime,
                  native_speculative_counters=counters,
                  screen_4K1024_only=True, observer_overhead_unmeasured=True,
                  official_benchmark=False, fresh_Math_not_run=True, adopted=False)
    (root / 'local_download_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
