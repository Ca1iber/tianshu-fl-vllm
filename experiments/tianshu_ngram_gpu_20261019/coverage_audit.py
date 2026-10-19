# SPDX-License-Identifier: Apache-2.0
"""Observe native GPU ngram target dispatch without changing graph selection."""
from collections import Counter
import json


def audit_coverage(root, state):
    read = lambda p: json.loads(p.read_text())
    directory = root / 'coverage_records'
    installed = [read(p) for p in directory.glob('installed_*.json')]
    assert installed
    prefix = '/workspace/vllm-plugin-FL/experiments/tianshu_ngram_gpu_20261019/'
    for record in installed:
        assert record['before_profile_capture_and_requests'] and record['native_GPU_ngram']
        assert not record['model_graph_dispatch_modified'] and not record['numeric_callbacks_installed']
        assert record['hook_sha256'] == state['frozen_sources'][prefix + 'hooks/sitecustomize.py']
        assert record['ngram_hook_sha256'] == state['frozen_sources'][prefix + 'hooks/ngram_sitecustomize.py']
    files = [read(p) for p in directory.glob('coverage_*.json')]
    assert len(files) == 1
    data = files[0]
    assert not data['errors'] and not data['changed_dispatch'] and not data['synchronized_GPU']
    assert {'round_1', 'round_2', 'round_3'} <= set(data['sealed_labels'])
    assert data['calls'] == sum(row['count'] for row in data['rows'])
    counts = Counter()
    for row in data['rows']:
        frame, count = row['frame'], row['count']
        if frame['label'] not in ('round_1', 'round_2', 'round_3'):
            continue
        assert count > 0 and 'error' not in frame
        if not frame['scheduled_tokens']:
            assert not frame['dispatches'] and not frame['graph_calls']
            continue
        assert len(frame['dispatches']) == len(frame['graph_calls']) == 1
        dispatch, call = frame['dispatches'][0], frame['graph_calls'][0]
        assert dispatch['mode'] == call['context_mode']
        assert (dispatch['mode'], call['branch']) in (
            ('NONE', 'eager_passthrough'), ('FULL', 'existing_graph_replay'))
        histogram = {int(k): v for k, v in dispatch['query_histogram'].items()}
        assert sum(histogram.values()) == dispatch['requests']
        assert sum(k * v for k, v in histogram.items()) == dispatch['tokens']
        counts[(frame['label'], dispatch['stage'], dispatch['mode'], call['branch'], dispatch['descriptor_uniform'])] += count
    proposer = [read(p) for p in (root / 'ngram_records').glob('proposer_*.json')]
    assert proposer and all(r['calls'] > 0 and not r['host_synchronization_added'] for r in proposer)
    return dict(complete=True, native_model_graph_dispatch_preserved=True,
                counts=[dict(label=k[0], stage=k[1], mode=k[2], graph_branch=k[3], uniform=k[4], count=v)
                        for k, v in sorted(counts.items(), key=lambda item: str(item[0]))],
                numeric_callbacks_installed=False, observer_GPU_synchronization=False,
                scope='Native FL host replay branch; not GPU kernel timing or a numerical oracle')
