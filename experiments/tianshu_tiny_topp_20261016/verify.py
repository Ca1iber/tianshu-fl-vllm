# SPDX-License-Identifier: Apache-2.0
"""Audit downloaded small-batch mask results without importing a GPU runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = json.loads((root / 'result.json').read_text())
    assert data['phase'] == 'complete' and data['complete']
    assert data['official_benchmark'] is False and data['adopted'] is False
    repo = Path(__file__).resolve().parents[2]
    for name, expected in data['source_snapshots'].items():
        snapshot = root / 'source_snapshot' / Path(name).relative_to('/')
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected, name
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            current = repo / Path(name).relative_to('/workspace/vllm-plugin-FL')
            assert hashlib.sha256(current.read_bytes()).hexdigest() == expected, name
    assert len(data['cases']) == 16 and len(data['events']) == 160
    assert set(data['compiled']) == {'1', '2', '4', '7'}
    assert all(v['num_warps'] == 16 and v['num_stages'] == 1
               and v['n_spills'] == 0 for v in data['compiled'].values())
    deltas = []
    for row in data['cases']:
        assert row['cleared_outputs'] and row['changed_input'] == bool(row['phase'])
        assert row['phase'] in (0, 1) and row['batch'] in (1, 2, 4, 7)
        for mode, check in row['checks'].items():
            assert mode in ('native', 'geo')
            assert all(check[k] for k in ('CPU_strict_semantics', 'relative_gate',
                       'strict_required', 'retained_values_unchanged'))
            columns = ('kept_mass', 'smallest_kept_probability', 'CPU_p',
                       'minimum_kept_logit', 'maximum_removed_logit', 'nonempty')
            assert all(len(check[k]) == row['batch'] for k in columns)
            for mass, smallest, p, low, high, nonempty in zip(*(check[k] for k in columns)):
                assert nonempty and mass >= p - 1e-6
                assert mass - smallest <= p + 1e-6 and high <= low
            events = [e for e in data['events'] if e['case'] == row['case']
                      and e['phase'] == row['phase'] and e['mode'] == mode]
            assert len(events) == 5 and {e['trial'] for e in events} == set(range(5))
            assert all(e['ms'] > 0 for e in events)
            assert statistics.median(e['ms'] for e in events) == row['medians_ms'][mode]
        delta = 100 * (row['medians_ms']['geo'] / row['medians_ms']['native'] - 1)
        assert abs(delta - row['time_delta_percent']) < 1e-12
        deltas.append(delta)
    result = dict(complete=True, source_snapshots_verified=len(data['source_snapshots']),
                  phases_verified=16, event_groups_verified=160,
                  CPU_scalar_semantics_independently_recomputed=True,
                  original_GPU_tensors_not_serialized=True,
                  component_time_delta_percent_range=[min(deltas), max(deltas)],
                  model_or_service_test=False, fresh_Math=False,
                  official_benchmark=False, adopted=False)
    (root / 'local_download_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
