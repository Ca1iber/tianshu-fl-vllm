# SPDX-License-Identifier: Apache-2.0
"""Recompute measured-phase graph coverage from complete sealed host counters."""
from collections import Counter
import json
from pathlib import Path


def audit_coverage(root, state):
    directory = root / 'coverage_records'
    installed = [json.loads(p.read_text()) for p in directory.glob('installed_*.json')]
    assert installed and all(r['before_profile_capture_and_requests'] for r in installed)
    prefix = '/workspace/vllm-plugin-FL/experiments/tianshu_spec_graph_coverage_20261019/'
    for r in installed:
        assert r['runner_source_sha256'] == 'b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'
        assert r['graph_source_sha256'] == 'ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2'
        assert r['hook_source_sha256'] == state['frozen_sources'][prefix + 'hooks/sitecustomize.py']
        assert r['observer_source_sha256'] == state['frozen_sources'][prefix + 'route_observer.py']
        assert not r['GPU_events_or_synchronizations_added']
    records = [json.loads(p.read_text()) for p in directory.glob('coverage_*.json')]
    assert len(records) == 1, 'Single GPU/single model-runner diagnostic expected'
    counts = Counter()
    phase_calls = Counter()
    data = records[0]
    assert not data['errors'] and not data['changed_dispatch'] and not data['synchronized_GPU']
    assert not data['overhead_measured']
    assert {'round_1', 'round_2'} <= set(data['sealed_labels'])
    assert data['calls'] == sum(row['count'] for row in data['rows'])
    assert data['outside_execute'].get('graph:capture', 0) > 0
    for row in data['rows']:
        f = row['frame']
        count = row['count']
        assert count > 0 and 'error' not in f
        hist = {int(k): v for k, v in f['scheduled_query_histogram'].items()}
        assert sum(hist.values()) == f['scheduled_requests']
        assert sum(k * v for k, v in hist.items()) == f['scheduled_tokens']
        if f['label'] not in ('round_1', 'round_2'):
            continue
        phase_calls[f['label']] += count
        if f['scheduled_tokens'] == 0:
            assert not f['dispatches'] and not f['graph_calls']
            counts[(f['label'], 'empty', 'NONE', 'no_forward')] += count
            continue
        assert len(f['dispatches']) == len(f['graph_calls']) == 1
        dispatch, call = f['dispatches'][0], f['graph_calls'][0]
        assert dispatch['tokens'] == f['scheduled_tokens']
        assert dispatch['requests'] == f['scheduled_requests']
        assert {int(k): v for k, v in dispatch['query_histogram'].items()} == hist
        assert dispatch['max_query'] == max(hist)
        assert dispatch['padded_tokens'] >= dispatch['tokens']
        assert dispatch['uniform_query_length'] == 2
        assert dispatch['force_uniform'] is None and not dispatch['force_eager']
        assert dispatch['mode'] == call['context_mode']
        assert call['wrapper_mode'] == 'FULL'
        assert (dispatch['mode'], call['branch']) in (
            ('NONE', 'eager_passthrough'), ('FULL', 'existing_graph_replay'))
        if dispatch['stage'] == 'prefill_or_mixed':
            assert dispatch['unfinished_prompt_requests'] > 0
            category = 'prefill_or_mixed'
        else:
            assert dispatch['stage'] == 'decode_or_verify' and dispatch['unfinished_prompt_requests'] == 0
            category = ('all_Q1' if set(hist) == {1} else 'all_Q2' if set(hist) == {2}
                        else 'mixed_Q1_Q2' if set(hist) == {1, 2} else 'other_decode')
        counts[(f['label'], category, dispatch['mode'], call['branch'])] += count
    assert set(phase_calls) == {'round_1', 'round_2'} and all(v > 0 for v in phase_calls.values())
    return dict(complete=True, sealed_rounds=['round_1', 'round_2'],
                actual_request_execute_calls=dict(phase_calls),
                counts=[dict(label=k[0], category=k[1], mode=k[2], graph_branch=k[3], count=v)
                        for k, v in sorted(counts.items())],
                outside_execute_counts=data['outside_execute'],
                scope='Successful pinned FL host replay API branch, not GPU per-kernel trace or timing',
                observer_overhead_unmeasured=True, official_benchmark=False, adopted=False)
