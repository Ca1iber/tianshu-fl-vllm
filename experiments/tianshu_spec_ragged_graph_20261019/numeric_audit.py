# SPDX-License-Identifier: Apache-2.0
"""Audit downloaded metadata/scalars; large GPU hidden/logits were not serialized."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from ragged_policy import GRAPH_PLANS, render_dummy


def audit_numeric(root, state):
    original_root = Path(state['output_root'])
    def read(path):
        return json.loads(path.read_text())
    def artifact(path):
        return root / Path(path).relative_to(original_root)
    raw_runner = root/'source_snapshot/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py'
    text, derivation = render_dummy(raw_runner.read_bytes())
    policies = [read(p) for p in (root/'ragged_records').glob('policy_*.json')]
    assert policies and any(p['runtime_calls'] > 0 for p in policies)
    expected = [dict(tokens=t, requests=n, uniform=False) for n, t in GRAPH_PLANS]
    for p in policies:
        assert p['registered_keys'] == expected and p['native_dispatcher_key_set_unchanged']
        assert p['derivation'] == derivation and artifact(p['generated_source']).read_text() == text
        assert p['dummy_capture_calls'] >= 42 and p['dummy_warm_calls'] >= 42
        assert all(tuple(map(int, name.split('x'))) in GRAPH_PLANS for name in p['runtime_plans'])
    files = [read(p) for p in (root/'ragged_records/numeric').glob('numeric_*.json')]
    active = [v for v in files if v['cases'] or v['errors']]
    assert len(active) == 1
    data = active[0]
    assert data['complete'] and not data['errors'] and 'pending_case' not in data
    assert data['observer_has_synchronization_and_extra_forward'] and not data['performance_score']
    cases = data['cases']
    assert len(cases) == 8 and len({r['input_sha256'] for r in cases}) == 8
    assert Counter(c['group'] for c in cases) == dict(exact_requests=3, padded_requests=3, equal_capacity=2)
    assert data['quotas'] == dict(exact_requests=3, padded_requests=3, equal_capacity=2)
    for r in cases:
        n, N, q, T = (r[k] for k in ('actual_requests','request_capacity','actual_queries','query_capacity'))
        assert (N, T) in GRAPH_PLANS and 16 <= n <= N <= 64 and q <= T and T*8 <= q*9
        assert r['padding_cu_seq_slots_verified'] and r['positions_and_block_table_slots_verified']
        assert len(r['query_lengths']) == n and set(r['query_lengths']) == {1, 2}
        assert len(r['cu']) == N+1 and r['cu'][0] == 0 and sum(r['query_lengths']) == q
        assert [r['cu'][i+1]-r['cu'][i] for i in range(n)] == r['query_lengths']
        assert all(v == q for v in r['cu'][n:])
        assert len(r['sequence_lengths']) == N and all(v == 0 for v in r['sequence_lengths'][n:])
        assert all(4096 <= v <= 131072 for v in r['sequence_lengths'][:n])
        assert len(r['slots']) == T and len(set(r['slots'][:q])) == q
        assert all(v >= 0 for v in r['slots'][:q]) and all(v == -1 for v in r['slots'][q:])
        assert len(r['positions']) == len(r['input_ids']) == len(r['mapped_KV_blocks']) == q
        assert all(0 <= v < 130560 for v in r['input_ids'])
        assert all(0 <= v < 131072 for v in r['positions'])
        for i, length in enumerate(r['query_lengths']):
            assert r['positions'][r['cu'][i]:r['cu'][i+1]] == list(range(r['sequence_lengths'][i]-length,r['sequence_lengths'][i]))
        assert [b*16+p%16 for b,p in zip(r['mapped_KV_blocks'],r['positions'])] == r['slots'][:q]
        assert len(r['KV_cache_shapes']) == len(r['KV_cache_strides']) == 42
        assert all(shape[1:] == [2,16,2,128] and max(r['mapped_KV_blocks']) < shape[0] for shape in r['KV_cache_shapes'])
        assert r['touched_KV_blocks'] == len(set(r['mapped_KV_blocks']))
        assert r['all42_candidate_KV_pages_restored'] and r['all42_untargeted_guard_pages_unchanged']
        assert r['RNG_unchanged'] and re.fullmatch('[0-9a-f]{64}',r['RNG_before_sha256'])
        assert r['RNG_before_sha256'] == r['RNG_after_sha256']
        assert not r['full_hidden_and_logits_serialized']
        if r['group'] == 'exact_requests':
            assert n == N and T > N
        elif r['group'] == 'padded_requests':
            assert n < N and T > N
        else:
            assert n < N and T == N
        expected_hash = hashlib.sha256(json.dumps({k:r[k] for k in
            ('cu','sequence_lengths','input_ids','positions')},sort_keys=True).encode()).hexdigest()
        assert r['input_sha256'] == expected_hash
        assert set(r['comparisons']) == {'hidden_padded','hidden_unpadded','logits_padded','logits_unpadded'}
        for c in r['comparisons'].values():
            assert c['passed'] and c['finite'] and c['bad_elements'] == 0
            assert c['atol'] == .02 and c['rtol'] == .01 and 0 <= c['max_tolerance_ratio'] <= 1
            assert c['max_abs'] >= 0
    assert any(a['query_capacity'] == b['query_capacity'] and a['request_capacity'] == b['request_capacity']
        and a['input_sha256'] != b['input_sha256'] for i,a in enumerate(cases) for b in cases[i+1:])
    assert state['profile_environment']['TIANSHU_RAGGED_NUMERIC_MODE'] == '1'
    return dict(complete=True, model_numeric_cases=8, changed_input_replay_verified=True,
        padded_and_fully_unpadded_eager_hidden_logits_pass=True, candidate_KV_restored_and_guards_unchanged=True,
        actual_GPU_metadata_slots_positions_checked=True, RNG_unchanged=True,
        owned42_capture_keys_and_actual_runtime_route_checked=True,
        scope='Downloaded metadata/scalar reductions audited; full GPU hidden/logits not serialized',
        diagnostic_only=True, official_benchmark=False, fresh_Math_not_run=True, adopted=False)
