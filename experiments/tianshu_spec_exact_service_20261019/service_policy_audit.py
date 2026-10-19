# SPDX-License-Identifier: Apache-2.0
"""Verify qualified exact-request policy and absence of numerical callbacks."""
import importlib.util
import json
from pathlib import Path


def audit_policy(root, state):
    read = lambda p: json.loads(p.read_text())
    base = Path(__file__).resolve().parent
    policy_path = base.parent / 'tianshu_spec_exact_requests_20261019/ragged_policy.py'
    spec = importlib.util.spec_from_file_location('_exact_service_policy_audit', policy_path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    gate = read(base / 'NUMERIC_GATE_32g.json')
    assert gate['complete'] and gate['model_numeric_cases'] == 8
    assert gate['request_count_padding'] is False
    name = '/workspace/vllm-plugin-FL/experiments/tianshu_spec_exact_requests_20261019/ragged_policy.py'
    assert state['frozen_sources'][name] == gate['frozen_sources'][name]
    directory = root / 'ragged_records'
    assert not (directory / 'numeric').exists(), 'Numerical callbacks must not run during service timing'
    assert 'TIANSHU_RAGGED_NUMERIC_MODE' not in state['profile_environment']
    assert state['profile_environment']['TIANSHU_RAGGED_SERVICE_MODE'] == state['mode']
    installed = [read(p) for p in (directory / 'coverage').glob('installed_*.json')]
    assert installed
    for item in installed:
        assert item['service_mode'] == state['mode']
        assert item['numeric_callbacks_installed'] is False
        assert item['policy_source_sha256'] == state['frozen_sources'][name]
    records = [read(p) for p in directory.glob('policy_*.json')]
    if state['mode'] == 'control':
        assert not records
    else:
        assert len(records) == 1
        record = records[0]
        expected = [dict(tokens=t, requests=n, uniform=False) for n, t in policy.GRAPH_PLANS]
        assert len(expected) == 248 and record['registered_keys'] == expected
        assert record['native_dispatcher_key_set_unchanged']
        assert record['dummy_warm_calls'] >= 248 and record['dummy_capture_calls'] >= 248
        assert record['runtime_calls'] > 0
        assert record['real_rows'] <= record['padded_rows']
        assert all(tuple(map(int, key.split('x'))) in policy.GRAPH_PLANS for key in record['runtime_plans'])
        runner = root / 'source_snapshot/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py'
        text, derivation = policy.render_dummy(runner.read_bytes())
        generated = root / Path(record['generated_source']).relative_to(state['output_root'])
        assert generated.read_text() == text and record['derivation'] == derivation
    return dict(complete=True, mode=state['mode'], qualified_numeric_cases=8,
                request_count_padding=False, numeric_callbacks_installed=False,
                native_graph_keys_preserved=True, owned_graph_keys=248 if records else 0,
                performance_measurement_has_no_extra_reference_forward=True)
