# SPDX-License-Identifier: Apache-2.0
"""Audit all warmed entries, real sampling/state arrays and no postwarm tuning."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
r = a.root.resolve()
d = json.loads((r/'result.json').read_text())
assert d['complete'] and d['phase'] == 'complete' and not d['adopted']
for path, sha in d['source_snapshots'].items():
    assert hashlib.sha256((r/'source_snapshot'/Path(path).relative_to('/')).read_bytes()).hexdigest() == sha
assert d['observed_libcache_db_url'] == 'sqlite:///'+d['isolated_cold_tuning_database']
assert d['observed_FlagGems_cache_dir'] == d['isolated_cache']
assert d['cold_warm_rng_before'] == d['cold_warm_rng_after']
assert d['real_failure_rng']['before'] == d['real_failure_rng']['after']
assert d['real_failure_rng']['original_exception_identity_preserved']
cold = [row for row in d['producer_records'] if row['phase'] == 'cold_warm']
keys = {row['entry_key'] for row in cold}
assert len(cold) == len(keys) == 4 and {row['N'] for row in cold} == {8355840}
expected = {(seed, offset) for seed in (7,2**32+7) for offset in (0,2**32)}
assert {(row['philox_seed'],row['philox_offset']) for row in cold} == expected
post = [row for row in d['producer_records'] if row['phase'] == 'postwarm']
assert len(post) == 64 and {row['batch'] for row in post} == set(range(1,65))
for row in d['producer_records']:
    assert row['entry_key'] in keys and row['N'] == row['batch']*130560
    assert row['BLOCK'] in (64,128,256,512) and row['warps'] in (4,8) and row['stages'] in (2,3)
    assert len(row['kernel_hash']) >= 32
assert len(d['warm_calls']) == 58
for warm in d['warm_calls']:
    assert warm['complete'] and warm['maximum_noise_bytes'] == 33423360
    assert {(c['seed'],c['offset']) for c in warm['calls']} == expected
    assert all(c['batch'] == 64 and c['elements'] == 8355840 for c in warm['calls'])
assert len(d['sampling_cases']) == 33 and len(d['scalar_boundary_cases']) == 24
assert {(c['batch'],c['generator_mode']) for c in d['sampling_cases']} == {
    (n,m) for n in (1,2,3,7,8,16,24,32,62,63,64) for m in ('global','mixed','all_seeded')}
assert {(c['seed'],c['offset'],c['batch']) for c in d['scalar_boundary_cases']} == {
    (seed,offset,n) for seed in (7,2**32+7) for offset in (0,2**31-4,2**31,2**32) for n in (1,8,64)}
tokens = 0
for c in d['sampling_cases'] + d['scalar_boundary_cases']:
    assert c['expected_tokens'] == c['actual_tokens'] and len(c['actual_tokens']) == c['batch']
    assert c['initial_default'] == c['after_warm_default']
    assert c['expected_final_default'] == c['actual_final_default']
    if 'generator_mode' in c:
        assert c['initial_per_request'] == c['after_warm_per_request']
        assert c['expected_final_per_request'] == c['actual_final_per_request']
    tokens += c['batch']
log = (r.parent/'probe.log').read_text()
assert log.count('COLD_WARM_BEGIN') == log.count('COLD_WARM_END') == log.count('POSTWARM_BEGIN') == log.count('POSTWARM_END') == 1
postlog = log.split('POSTWARM_BEGIN',1)[1].split('POSTWARM_END',1)[0]
assert not any('fused_exponential_kernel_f32_unroll8' in line and
    ('Autotuning kernel' in line or 'Triton autotuning for function' in line) for line in postlog.splitlines())
out = dict(complete=True, component_only=True, all_four_actual_scalar_type_entries_warmed=True,
    all64_postwarm_shapes_and_all_sampling_zero_new_exponential_tuning=True,
    sampling_cases=57, exact_sample_tokens=tokens, actual_entry_keys_and_compiled_configs_checked=True,
    default_and_request_rng_states_and_real_exception_restore_checked=True,
    sources=len(d['source_snapshots']), cold_warm_seconds=d['warm_calls'][0]['elapsed_seconds'],
    cold_native_first_configuration_sample_equivalence_not_asserted=True,
    new_service_and_Math_not_run=True, adopted=False)
(r/'local_download_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
