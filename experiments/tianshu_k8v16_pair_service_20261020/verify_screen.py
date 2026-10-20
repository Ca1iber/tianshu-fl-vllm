# SPDX-License-Identifier: Apache-2.0
"""Independent request, source, runtime-binding and timing audit for paired K8."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'tianshu_k8v16_pair_service_20261020'))
from audit_profile import audit_runtime


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def success(path):
    result = {}
    for line in path.read_text().splitlines():
        if line.startswith('vllm:request_success_total'):
            reason = re.search(r'finished_reason="([^"]+)"', line)[1]
            result[reason] = result.get(reason, 0) + float(line.rsplit(' ', 1)[1])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--label', choices=('4K', '16K'), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert root.is_relative_to('/workspace/benchmark-results')
    ilen = 4096 if args.label == '4K' else 16384
    states = {mode: read(root / mode / 'result.json') for mode in ('control', 'candidate')}
    control, candidate = states.values()
    assert control['frozen_sources'] == candidate['frozen_sources']
    expected_count=read(BASE/'PROFILE_CPU_GATE_32g.json')['frozen_source_count']
    assert len(control['frozen_sources']) == expected_count
    assert control['server_command'] == candidate['server_command']
    assert control['server_command'] == ['vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031', '--compilation-config', '{"cudagraph_mode": "FULL_DECODE_ONLY"}', '--served-model-name', 'minicpm', '--gpu-memory-utilization', '0.85', '--max-model-len', '131072']
    assert len(control['profile_environment']) == len(candidate['profile_environment']) == 54
    assert control['profile_environment'].keys() == candidate['profile_environment'].keys()
    for key, value in control['profile_environment'].items():
        other = candidate['profile_environment'][key]
        if key in ('TIANSHU_PAIR_LOAD_MODE',):
            assert (value, other) == ('control', 'candidate')
        else:
            assert value.replace(str(root / 'control'), '<RUN>') == other.replace(str(root / 'candidate'), '<RUN>'), key
    result = dict(complete=False, official_benchmark=False, adopted=False,
        fresh_Math_not_run=True, quality_pending=True,
        cross_configuration_text_is_diagnostic=True,
        exact_cross_configuration_text_is_not_an_accuracy_measure=True,
        frozen_sources=control['frozen_sources'], frozen_source_count=expected_count,
        profile_environment_count=54, original_serve_command=control['server_command'],
        input_tokens=ilen, output_tokens=1024, requests_per_round=64,
        rounds_per_mode=3, excluded_rounds=[1], modes={})
    for mode, state in states.items():
        arm = root / mode
        assert state['phase'] == 'complete' and state['mode'] == mode
        assert state['workload'] == dict(input_tokens=ilen, output_tokens=1024, concurrency=64,
            requests_per_round=64, rounds=3, excluded_rounds=[1])
        assert len(state['screen_rounds']) == 3
        audit = audit_runtime(arm, state, official=(ilen == 4096))
        assert audit['actual_FL_Graph_capture_and_replay_verified']
        assert audit['actual_arena_reserved_inside_memory_profiling_verified']
        assert audit['arena_capacity_pages'] == 8192
        assert audit['native_scheduler_limit_verified']
        assert audit['max_num_seqs'] == 64
        assert audit['native_watermark_blocks'] == 0
        assert audit['pair_load_binding_verified'] and audit['actual_B64_capture_and_model_replay_verified']
        expected_decode='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002' if mode=='candidate' else '7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
        assert audit['active_decode_source_sha256']==expected_decode
        rounds, seeds = [], []
        for index, row in enumerate(state['screen_rounds'], 1):
            assert row['round'] == index and row['successful_requests'] == len(row['responses']) == 64
            assert math.isfinite(row['duration_seconds']) and row['duration_seconds'] > 0
            sample_seeds = []
            for response in row['responses']:
                assert response['usage'] == dict(prompt_tokens=ilen, completion_tokens=1024, total_tokens=ilen + 1024)
                assert response['input_tokens'] == ilen and response['output_tokens'] == 1024
                assert response['text'] and response['api_seed'] is None
                assert math.isfinite(response['ttft_seconds']) and response['ttft_seconds'] > 0
                sample_seeds.append(response['input_seed'])
            assert sample_seeds == [202610203000 + 10 * ilen + 100 * (index - 1) + i for i in range(64)]
            seeds.append(sample_seeds)
            total = 64 * (ilen + 1024) / row['duration_seconds']
            assert abs(total - row['total_tokens_per_second']) < 1e-10
            rounds.append({k: v for k, v in row.items() if k != 'responses'})
        before, after = success(arm / 'before_metrics.txt'), success(arm / 'after_metrics.txt')
        assert after.get('error', 0) == after.get('abort', 0) == 0
        assert sum(after.values()) - sum(before.values()) == 192
        canaries = state['seeded_canaries']
        assert len(canaries) == 2 and canaries[0]['text'] == canaries[1]['text']
        for c in canaries:
            assert c['input_seed'] == 2026101699 and c['api_seed'] == 777
            assert c['usage'] == dict(prompt_tokens=128, completion_tokens=16, total_tokens=144)
        steady = [r['total_tokens_per_second'] for r in rounds[1:]]
        assert statistics.mean(steady) == state['mean_total_tok_s']
        result['modes'][mode] = dict(rounds=rounds, round_input_seeds=seeds,
            mean_total_tok_s=state['mean_total_tok_s'], spread_percent=state['spread_percent'],
            mean_TTFT_seconds=statistics.mean(r['mean_ttft_seconds'] for r in rounds[1:]),
            profile_audit=audit, profile_environment=state['profile_environment'],
            seeded_canary_sha256=hashlib.sha256(canaries[0]['text'].encode()).hexdigest(),
            seeded_canary_repeatable_within_configuration=True,
            metrics_success_before=before, metrics_success_after=after,
            original_result_path=str(arm / 'result.json'), original_result_sha256=sha(arm / 'result.json'))
    a, b = result['modes'].values()
    assert a['round_input_seeds'] == b['round_input_seeds']
    assert len({seed for row in a['round_input_seeds'] for seed in row}) == 192
    result.update(complete=True, independently_rechecked=True, exact_screen_requests_verified=384,
        cross_configuration_seeded_text_equal=a['seeded_canary_sha256'] == b['seeded_canary_sha256'],
        control_total_tok_s=a['mean_total_tok_s'], candidate_total_tok_s=b['mean_total_tok_s'],
        gain_percent=100 * (b['mean_total_tok_s'] / a['mean_total_tok_s'] - 1),
        no_full_responses_or_tensor_data_in_key_result=True, analyzer_source_sha256=sha(__file__))
    original_state = root / 'state.json'
    if original_state.exists():
        original = read(original_state)
        result['original_pair_state'] = {k: original.get(k) for k in ('phase', 'error', 'runner_pid')}
        result['original_pair_state_sha256'] = sha(original_state)
    output = BASE / f'DIAGNOSTIC_{args.label}_GATE_32g.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (root/'independent_screen_gate.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (BASE / f'DIAGNOSTIC_{args.label}_ROUNDS_32g.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'round', 'total_tok_s', 'output_tok_s', 'TTFT_ms', 'duration_s', 'success', 'excluded'])
        for mode, part in result['modes'].items():
            for row in part['rounds']:
                writer.writerow([mode, row['round'], row['total_tokens_per_second'], row['output_tokens_per_second'], row['mean_ttft_seconds'] * 1000, row['duration_seconds'], 64, row['round'] == 1])
    print(json.dumps({k: result[k] for k in ('complete', 'control_total_tok_s', 'candidate_total_tok_s', 'gain_percent', 'cross_configuration_seeded_text_equal', 'quality_pending')}))


if __name__ == '__main__':
    main()
