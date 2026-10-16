# SPDX-License-Identifier: Apache-2.0
"""Independently recompute original BlockTable cost records after download."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    state = json.loads((root / 'result.json').read_text())
    assert state['phase'] == 'complete' and state['complete']
    assert not state['official_benchmark'] and not state['adopted']
    assert state['original_methods_unmodified'] and state['full_model_not_run']
    assert state['record_oplist'] and state['record_once']
    repo = Path(__file__).resolve().parents[2]
    assert len(state['source_snapshots']) == 4
    for name, expected in state['source_snapshots'].items():
        snapshot = root / 'source_snapshot' / Path(name).relative_to('/')
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected, name
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            local = repo / Path(name).relative_to('/workspace/vllm-plugin-FL')
            assert hashlib.sha256(local.read_bytes()).hexdigest() == expected, name
    assert len(state['cases']) == 6
    cases = {(c['columns'], c['requests']): c for c in state['cases']}
    assert set(cases) == {(col, n) for col in (8192, 320) for n in (1, 8, 64)}
    medians = {}
    for key, case in cases.items():
        columns, n = key
        assert case['table_shape'] == [256, columns]
        assert case['active_columns'] == 320
        assert case['transfer_bytes'] == n * columns * 4 and case['pinned']
        assert case['positions_dtype'] == case['slot_dtype'] == 'torch.int64'
        phases = case['changed_graph_checks']
        assert [p['step'] for p in phases] == [0, 1]
        for p in phases:
            assert p['outputs_cleared']
            assert p['checks'] == dict(slots_and_padding=True, full_active_rows_and_tail=True)
        samples = case['timings']
        assert len(samples) == 21
        assert {(s['repeat'], s['name']) for s in samples} == {
            (r, name) for r in range(7) for name in ('commit', 'slots', 'commit_and_slots')}
        for s in samples:
            assert s['iterations'] == 20 and s['host_wall_ms'] > 0 and s['gpu_stream_ms'] > 0
        actual = {name: {metric: statistics.median(s[metric] for s in samples if s['name'] == name)
                        for metric in ('host_wall_ms', 'gpu_stream_ms')}
                  for name in ('commit', 'slots', 'commit_and_slots')}
        assert actual == case['medians']
        medians[f'columns{columns}_B{n}'] = actual
    audit = dict(complete=True, source_snapshots_verified=4, cases_verified=6,
                 changed_Graph_check_records_verified=12, raw_timing_records_recomputed=126,
                 full_GPU_tensor_values_not_serialized=True,
                 model_speed_and_quality_not_measured=True, adopted=False, medians=medians)
    (root / 'local_download_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
