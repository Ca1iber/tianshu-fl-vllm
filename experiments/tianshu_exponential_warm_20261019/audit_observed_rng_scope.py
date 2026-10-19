# SPDX-License-Identifier: Apache-2.0
"""Preserve the failed stability gate and verify the actually observed RNG scope."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
d = json.loads((root / 'result.json').read_text())
assert d['phase'] == 'complete' and d['complete'] and not d['adopted']
for path, expected in d['source_snapshots'].items():
    assert hashlib.sha256((root / 'source_snapshot' / Path(path).relative_to('/')).read_bytes()).hexdigest() == expected
assert d['observed_libcache_db_url'] == 'sqlite:///' + d['isolated_cold_tuning_database']
assert d['observed_FlagGems_cache_dir'] == d['isolated_cache']
assert d['cold_warm_rng_before'] == d['cold_warm_rng_after']
assert d['real_failure_rng']['before'] == d['real_failure_rng']['after']
assert d['real_failure_rng']['original_exception_identity_preserved']
records = d['producer_records']
for phase in ('cold_warm', 'postwarm'):
    rows = [r for r in records if r['phase'] == phase]
    assert len(rows) == 64 and {r['batch'] for r in rows} == set(range(1, 65))
    assert all(r['N'] == r['batch'] * 130560 for r in rows)
for r in records:
    assert r['N'] == r['batch'] * 130560 and r['BLOCK'] in (64, 128, 256, 512)
    assert r['warps'] in (4, 8) and r['stages'] in (2, 3) and len(r['kernel_hash']) >= 32
assert len(d['warm_calls']) == 34
for warm in d['warm_calls']:
    assert warm['complete'] and warm['maximum_noise_bytes'] == 33423360
    assert [r['batch'] for r in warm['shapes']] == list(range(64, 0, -1))
assert len(d['sampling_cases']) == 33
assert {(r['batch'], r['generator_mode']) for r in d['sampling_cases']} == {
    (n, mode) for n in (1, 2, 3, 7, 8, 16, 24, 32, 62, 63, 64)
    for mode in ('global', 'mixed', 'all_seeded')}
for r in d['sampling_cases']:
    assert r['expected_tokens'] == r['actual_tokens'] and len(r['actual_tokens']) == r['batch']
    assert r['initial_default'] == r['after_warm_default']
    assert r['initial_per_request'] == r['after_warm_per_request']
    assert r['expected_final_default'] == r['actual_final_default']
    assert r['expected_final_per_request'] == r['actual_final_per_request']
    assert set(r['actual_final_per_request']) == {str(i) for i in r['seeded_rows']}
log = (root.parent / 'probe.log').read_text()
cold = log.split('COLD_WARM_BEGIN', 1)[1].split('COLD_WARM_END', 1)[0]
post = log.split('POSTWARM_BEGIN', 1)[1].split('POSTWARM_END', 1)[0]
summaries = lambda text: [line for line in text.splitlines()
    if 'Triton autotuning for function fused_exponential_kernel_f32_unroll8 finished' in line]
assert len(summaries(cold)) == len(summaries(post)) == 1
assert "config key: (8355840, 'torch.float32')" in summaries(cold)[0]
assert "config key: (130560, 'torch.float32')" in summaries(post)[0]
seed = int(re.search(r'philox_seed=(\d+)', cold).group(1))
libpath = next(path for path in d['source_snapshots'] if path.endswith('/utils/libentry.py'))
tree = ast.parse((root / 'source_snapshot' / Path(libpath).relative_to('/')).read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LibEntry')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'key')
ns = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), libpath, 'exec'), ns)
owner = SimpleNamespace(divisibility=16)
native_key = lambda n, seed, offset: ns['key'](owner, [], [n, seed, offset], [])
assert native_key(8355840, seed, 0) == native_key(130560, seed, 0) == ('i32', 'i64', 'i32')
assert native_key(130560, 2026101901, 0) == ('i32', 'i32', 'i32')
assert native_key(130560, 2026101901, 2**32) == ('i32', 'i32', 'i64')
out = dict(complete_for_observed_rng_scope=True, original_strict_stability_gate_preserved=True,
    stability_claim_passed=False, original_cold_config_count_expected64_observed1=True,
    postwarm_new_exponential_tuning=1, source_snapshots=len(d['source_snapshots']),
    observed_all64_shapes_dispatched=True, actual_compiled_configurations_checked=True,
    sampling_cases=33, exact_sample_tokens=sum(r['batch'] for r in d['sampling_cases']),
    CPU_CUDA_and_per_request_rng_states_and_failure_restore_checked=True,
    actual_LibEntry_key_AST_executed=True, N_values_within_i32_share_entry_key=True,
    cold_seed_class='i64', sampling_seed_class='i32',
    offset_class_can_also_change_entry_key=True, cold_warm_seconds=d['warm_calls'][0]['elapsed_seconds'],
    standalone_component_only=True, new_service_and_Math_not_run=True, adopted=False)
(root / 'local_observed_rng_scope_audit.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))
