# SPDX-License-Identifier: Apache-2.0
"""Summarize complete counter experiments without treating closure as success."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.root / 'result.json').read_text())
    assert data.get('complete') is True
    histograms = data['histogram_cases']
    sorts = data['sort_cases']
    assert len(histograms) == 16 and len(sorts) == 120
    assert all(len(case['variants']) == 5 for case in histograms)
    summary = dict(complete=True, official_benchmark=False, adopted=False,
                   source_sha256=data['original_sort_sha256'], widths={},
                   installed_sources_modified=data['installed_sources_modified'])
    for width in (64, 32, 16, 8):
        selected = [next(row for row in case['variants'] if row['label'] == str(width))
                    for case in histograms]
        original = [next(row for row in case['variants'] if row['label'] == 'original64')
                    for case in histograms]
        reference = [next(row for row in case['variants'] if row['label'] == '64')
                     for case in histograms]
        sort_rows = [row for row in sorts if row['bits'] == width]
        histogram_ok = all(row.get('exact_cpu_histogram_passed') for row in selected + original + reference)
        sort_ok = all(row.get('actual_histogram_routed') and row.get('values_exact')
                      and row.get('stable_indices_exact') and row.get('indices_dtype') == 'torch.int64'
                      and row.get('graph_values_exact') and row.get('graph_stable_indices_exact')
                      and 'error' not in row for row in sort_rows)
        report = dict(histogram_cases_passed=sum(bool(row.get('exact_cpu_histogram_passed')) for row in selected),
                      sort_cases_passed=sum(bool(row.get('values_exact') and row.get('stable_indices_exact')
                                                and row.get('graph_values_exact') and row.get('graph_stable_indices_exact')
                                                and row.get('actual_histogram_routed') and 'error' not in row)
                                            for row in sort_rows),
                      numerical_checks_passed=histogram_ok and sort_ok,
                      actual_histogram_widths=sorted({row['actual_bits'] for row in selected}),
                      actual_model_shape_histograms=[], actual_model_shape_sorts=[])
        for case, row, direct, generated in zip(histograms, selected, original, reference):
            if (case['m'], case['n'], case['dtype'], case['pattern']) != (64, 130560, 'torch.float32', 'random'):
                continue
            entry = dict(descending=case['descending'], candidate=row)
            if all('median_ms' in x for x in (row, direct, generated)):
                entry.update(vs_original_speedup_percent=100 * (direct['median_ms'] / row['median_ms'] - 1),
                             vs_generated64_speedup_percent=100 * (generated['median_ms'] / row['median_ms'] - 1),
                             original_ms=direct['median_ms'], generated64_ms=generated['median_ms'])
            report['actual_model_shape_histograms'].append(entry)
        for row in sort_rows:
            if (row['m'], row['n'], row['pattern']) != (64, 130560, 'random'):
                continue
            baseline = next(x for x in sorts if x['bits'] == 64 and
                            all(x[k] == row[k] for k in ('m', 'n', 'pattern', 'descending')))
            entry = dict(descending=row['descending'], median_ms=row.get('median_sort_ms'),
                         generated64_ms=baseline.get('median_sort_ms'))
            if 'median_sort_ms' in row and 'median_sort_ms' in baseline:
                entry['vs_generated64_speedup_percent'] = 100 * (baseline['median_sort_ms'] / row['median_sort_ms'] - 1)
            report['actual_model_shape_sorts'].append(entry)
        summary['widths'][str(width)] = report
    (args.root / 'counter_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
