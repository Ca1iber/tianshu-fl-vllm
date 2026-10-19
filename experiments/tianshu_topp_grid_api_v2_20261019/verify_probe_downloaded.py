# SPDX-License-Identifier: Apache-2.0
"""Audit the downloaded real-API gate before enabling any model service."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--write-gate', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    data = json.loads((root / 'result.json').read_text())
    assert data['phase'] == 'complete' and data['complete']
    assert not data['model_test'] and not data['adopted'] and not data['fresh_Math']
    assert data['imported_rejection_alias_verified'] and data['sampler_forward_unchanged']
    repo = Path(__file__).resolve().parents[2]
    for name, expected in data['source_snapshots'].items():
        assert digest(root / 'source_snapshot' / Path(name).relative_to('/')) == expected, name
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert digest(repo / Path(name).relative_to('/workspace/vllm-plugin-FL')) == expected, name
    assert all(data['source_snapshots'][p] == h for p, h in data['profile_frozen_sources'].items())
    assert len(data['masks']) == 156 and len(data['events']) == 260
    expected_cases=[(b,'normal') for b in (32,33,40,48,56,63,64)]+[
        (64,'bf16_normal'),(64,'bf16_peaked'),(64,'ties'),(64,'equal'),(64,'masked'),(64,'one_finite')]
    identifiers = set()
    for row in data['masks']:
        key = row['case'], row['probability'], row['phase']
        assert key not in identifiers
        identifiers.add(key)
        assert row['case'] in range(13) and row['phase'] in (0, 1)
        assert (row['batch'],row['distribution'])==expected_cases[row['case']]
        assert row['probability'] in (.0001, .1, .5, .95, .999, 1.0)
        assert row['cleared_outputs'] and row['changed_input'] == bool(row['phase'])
        c = row['checks']['candidate']
        assert all(c[k] for k in ('relative_gate', 'strict_required', 'CPU_strict_semantics', 'retained_values_unchanged'))
        columns = ('kept_mass', 'smallest_kept_probability', 'CPU_p',
                   'minimum_kept_logit', 'maximum_removed_logit', 'nonempty')
        assert all(len(c[k]) == row['batch'] for k in columns)
        for mass, smallest, p, low, high, nonempty in zip(*(c[k] for k in columns)):
            assert nonempty and mass >= p - 1e-6
            assert mass - smallest <= p + 1e-6 and high <= low
            assert abs(p - row['probability']) < 1e-7
        if row['probability'] == .95:
            for mode in ('original', 'candidate'):
                events = [e for e in data['events'] if e['case'] == row['case']
                          and e['phase'] == row['phase'] and e['mode'] == mode]
                assert len(events) == 5 and {e['trial'] for e in events} == set(range(5))
                assert all(e['ms'] > 0 for e in events)
                assert statistics.median(e['ms'] for e in events) == row['medians_ms'][mode]
    assert len(data['RNG_cases']) == 21 and len(data['fallbacks']) == 8
    assert {(r['batch'], r['seeded']) for r in data['RNG_cases']} == {
        (b, s) for b in (32,33,40,48,56,63,64) for s in ('none', 'partial', 'all')}
    for row in data['RNG_cases']:
        assert row['global_and_request_states_identical'] and row['sampled_from_retained_logits']
        states = row['global_after_sha256']
        assert len(states['original']) == 64 and states['original'] == states['candidate']
        assert row['request_after_sha256']['original'] == row['request_after_sha256']['candidate']
        count = 0 if row['seeded'] == 'none' else row['batch'] - 1 if row['seeded'] == 'partial' else row['batch']
        assert len(row['request_after_sha256']['candidate']) == count
        assert all(len(v) == 64 for v in row['request_after_sha256']['candidate'].values())
        assert all(len(tokens) == row['batch'] for tokens in row['sampled_tokens'].values())
        equal = row['sampled_tokens']['original'] == row['sampled_tokens']['candidate']
        assert equal == row['identical_tokens']
        if row['identical_mask']:
            assert equal
    assert len(data['lifecycle_cases'])==6
    assert [row['batch'] for row in data['lifecycle_cases']]==[64,32,33,64,32,33]
    for row in data['lifecycle_cases']:
        assert all(row[k] for k in ('input_changed','same_graph_changed_input_exact',
            'older_other_batch_outputs_preserved','RNG_unchanged'))
        assert row['actual_buffer_rows']==64
    assert [(r['batch'],r['topk'],r['topp']) for r in data['fallbacks']]==[
        (8,False,True),(16,False,True),(24,False,True),(31,False,True),
        (65,False,True),(64,False,False),(64,True,False),(64,True,True)]
    assert all(r['bitwise'] for r in data['fallbacks'])
    report = dict(complete=True, masks=156, RNG_cases=21, fallback_cases=8, cross_batch_lifecycle_cases=6, event_groups=260,
                  source_snapshots=len(data['source_snapshots']),
                  scope='Downloaded scalar checks, source and RNG hashes audited; full GPU logits not serialized',
                  model_or_quality_test=False, adopted=False,
                  remote_result=data['output_root'] + '/result.json',
                  remote_result_sha256=digest(root / 'result.json'),
                  profile_frozen_sources=data['profile_frozen_sources'])
    (root / 'local_download_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.write_gate:
        assert not args.write_gate.exists()
        args.write_gate.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'profile_frozen_sources'}))


if __name__ == '__main__':
    main()
