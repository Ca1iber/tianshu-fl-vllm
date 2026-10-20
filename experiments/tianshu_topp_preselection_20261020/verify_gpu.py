# SPDX-License-Identifier: Apache-2.0
"""Recheck saved FP64 mass/order diagnostics, actual kernels and timings."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
EXPECTED = dict(geo='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d',
    shift_half='7811afa331e5179461da6b0c5747e5163a4ee4ab518a695233f405652d2a14b0',
    shift_one='52c1d34e35ad040d4101e4871408104c0471d57d64062497a49a9b4e0d2f49c8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
assert root.is_relative_to('/workspace/benchmark-results')
path = root / 'micro/result.json'
data = json.loads(path.read_text())
assert (root / 'complete.txt').is_file() and (root / 'exit_code.txt').read_text().strip() == '0'
assert data['complete'] and data['all_plans_pass_all_cases'] and data['same_Graph_RNG_checks']
assert data['relative_passing_plans'] == list(EXPECTED)
assert data['vocab_size'] == 130560 and data['CPU_tolerance'] == 1e-6
assert len(data['cases']) == 12
for name, digest in data['frozen_sources'].items():
    assert sha(name) == digest, name
    assert sha(root / 'micro/source_snapshot' / Path(name).relative_to('/')) == digest, name
rows = []
checks = 0
for index, case in enumerate(data['cases'], 1):
    assert len(case['variants']) == 3
    assert [r['label'] for r in case['variants']] == list(EXPECTED)
    baseline = case['variants'][0]['median_ms']
    assert case['P_changed_in_place'] == (case['distribution'] == 'varied_p')
    if case['P_changed_in_place']:
        assert case['input_p_before'][::-1] == case['input_p_after']
    for entry in case['variants']:
        assert 'error' not in entry and entry['original_graph_exact'] and entry['changed_graph_exact']
        meta = entry['compiled_metadata']
        expected_sha = data['native_source_sha256'] if case['topk'] else EXPECTED[entry['label']]
        assert meta['source_sha256'] == expected_sha == sha(meta['source'])
        assert meta['corrected_source_active'] == (not case['topk'])
        assert meta['num_warps'] == 16 and meta['num_stages'] == 1
        assert meta['BLOCK_SIZE'] == (8192 if case['topk'] else 32768)
        assert meta['BLOCK_SIZE_TRUNC'] == (4096 if case['topk'] else 16384)
        assert meta['grid'] == [min(case['batch'], 16)] and meta['VOCAB_SIZE'] == 130560
        assert meta['TOPK_ENABLED'] == case['topk'] and meta['TOPP_ENABLED']
        assert len(entry['raw_ms']) == 7 and all(math.isfinite(x) and x > 0 for x in entry['raw_ms'])
        assert statistics.median(entry['raw_ms']) == entry['median_ms']
        for phase, probabilities in [('original', case['input_p_before']), ('changed', case['input_p_after'])]:
            r = entry[phase]
            assert r['CPU_strict_semantics'] and r['relative_gate'] and r['strict_required'] and r['retained_values_unchanged']
            assert r['CPU_p'] == probabilities
            assert all(len(r[k]) == case['batch'] for k in ('CPU_p', 'kept_mass', 'smallest_kept_probability', 'minimum_kept_logit', 'maximum_removed_logit', 'nonempty'))
            for prob, mass, minimum, low, high, nonempty in zip(r['CPU_p'], r['kept_mass'], r['smallest_kept_probability'], r['minimum_kept_logit'], r['maximum_removed_logit'], r['nonempty']):
                assert nonempty and math.isfinite(mass) and math.isfinite(minimum)
                assert mass >= prob - 1e-6 and mass - minimum <= prob + 1e-6
                assert low is not None and (high is None or high <= low)
                checks += 1
        rows.append(dict(case=index, batch=case['batch'], distribution=case['distribution'], topk=case['topk'], label=entry['label'], median_ms=entry['median_ms'], throughput_gain_percent=100 * (baseline / entry['median_ms'] - 1), n_regs=meta['n_regs'], n_spills=meta['n_spills'], shared_bytes=meta['shared']))
gate = dict(complete=True, independently_rechecked=True, CPU_only=False, official_benchmark=False, adopted=False,
    numerical_checks_passed=True, GPU_cases=12, plans_per_case=3, GPU_plan_cases=36, input_phases=2,
    strict_row_mass_order_checks=checks, same_Graph_changed_logits_and_P_checks=True,
    RNG_unchanged_recorded_by_instrumented_probe=True, GPU_execution_not_repeated_by_offline_audit=True,
    timing_includes_identical_input_restore=True, source_sha256=EXPECTED,
    raw_result_path=str(path), raw_result_sha256=sha(path), analyzer_sha256=sha(__file__), frozen_sources=data['frozen_sources'],
    no_service_or_Math_for_negative_candidate=True, no_full_responses_or_tensors_in_key_result=True, rows=rows)
(BASE / 'GPU_GATE_32g.json').write_text(json.dumps(gate, indent=2, allow_nan=False) + '\n')
with (BASE / 'GPU_TIMINGS_32g.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print(json.dumps(dict(complete=True, plan_cases=36, strict_row_mass_order_checks=checks, adopted=False)))
for row in rows:
    if row['case'] in (3, 4, 9, 12):
        print(json.dumps(row))
