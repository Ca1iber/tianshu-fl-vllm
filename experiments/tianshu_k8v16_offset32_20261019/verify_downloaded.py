# SPDX-License-Identifier: Apache-2.0
"""Independently audit same-layout K8V16 offset32 from original artifacts."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from layout import MODES, geometry
from plans import PLANS, configuration

base = Path(__file__).resolve().parent
repo = base.parents[1]
p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
d = json.loads((root / 'result.json').read_text())
sources = d['source_snapshots']
for name, sha in sources.items():
    assert hashlib.sha256((root / 'source_snapshot' / Path(name).relative_to('/')).read_bytes()).hexdigest() == sha
    if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
        assert hashlib.sha256((repo / Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest() == sha
spec = importlib.util.spec_from_file_location('_mixed_verify_codec_parent', base.parent / 'tianshu_soft8_engine_v2_20261019/source.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
assert d['all65536_actual_writer_encoded'] == [parent.encode_bits(i) for i in range(65536)]
assert d['all256_decoded_bits'] == [(((v & 127) << 4) + 14848 if (v & 127) >= 8 else 0) | ((v & 128) << 8) for v in range(256)]
proof = d['codec_proof']
remote_root = Path(proof['writer_path']).parents[1]

def artifact(path):
    return root / Path(path).relative_to(remote_root)

cv = next(name for name, sha in sources.items() if sha == '8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289')
text, generated = parent.prior.render_source((root / 'source_snapshot' / Path(cv).relative_to('/')).read_bytes())
assert artifact(proof['reader']['generated_source']).read_text() == text
assert generated['generated_sha256'] == proof['reader']['generated_sha256']
assert hashlib.sha256(artifact(proof['writer_path']).read_bytes()).hexdigest() == proof['writer_sha256']
assert d['phase'] == 'complete' and d['complete'] and d['modes'] == list(MODES)
assert d['plans'] == {k:list(v) if v is not None else None for k,v in PLANS.items()}
assert not d['adopted'] and not d['official_benchmark']
cases = ((1, 33), (1, 4096), (8, 4096), (16, 4096), (24, 4096),
    (32, 4096), (64, 4096), (64, 5120))
assert [(c['batch'], c['context']) for c in d['cases']] == list(cases)
cpu_elements = cpu_refs = gpu_refs = event_groups = 0
timings = {}
for case_id, c in enumerate(d['cases']):
    n, context = c['batch'], c['context']
    assert c['strided'] == bool(case_id % 2) and c['timing_includes_real_writer']
    assert [x['mode'] for x in c['layouts']] == list(MODES)
    for layout in c['layouts']:
        g = geometry(layout['mode'])
        assert all(layout[k] == v for k, v in g.items() if k != 'mode')
        assert layout['storage_mode'] == 'K8V16'
        assert layout['actual_spec_page_bytes'] == g['page_bytes']
        assert layout['actual_raw_bytes'] == layout['physical_pages'] * g['page_bytes']
        stride = g['page_bytes'] * (2 if c['strided'] else 1)
        assert layout['actual_native_view_stride_bytes'] == stride
        assert layout['key_strides'] == [stride // g['key_bytes'], 256, 128, 1]
        assert layout['value_strides'] == [stride // g['value_bytes'], 256, 128, 1]
        assert layout['plane_aliases_actual_raw'] and layout['no_hidden_BF16_cache']
    lengths = [max(1, context - i * 37 % min(context, 257)) for i in range(n)]
    assert len(c['phases']) == 2
    for phase, entry in enumerate(c['phases']):
        if phase:
            lengths = [max(1, length - 7) for length in lengths]
        assert entry['phase'] == phase and entry['lengths'] == lengths
        assert entry['outputs_bitwise_same']
        assert entry['changed_input_KV_slots_and_lengths'] == bool(phase)
        assert [x['mode'] for x in entry['checks']] == list(MODES)
        for check in entry['checks']:
            assert all(check[k] for k in ('all_physical_storage_checked', 'output_and_scratch_guards', 'inputs_unchanged', 'RNG_unchanged'))
            ref = check['GPU_reference']
            assert ref['passed'] and ref['failed_elements'] == 0 and ref['atol'] == .02 and ref['rtol'] == .01
            gpu_refs += 1
            expected_pairs = {(i, h) for i in {0, n // 2, n - 1} for h in (0, 7, 8, 15)}
            assert {(r['sequence'], r['head']) for r in check['CPU_references']} == expected_pairs
            for r in check['CPU_references']:
                assert len(r['actual']) == len(r['expected']) == 128
                errors = [abs(x - y) for x, y in zip(r['actual'], r['expected'])]
                bad = sum(err > .02 + .01 * abs(y) for err, y in zip(errors, r['expected']))
                assert bad == r['failed_elements'] == 0 and r['passed']
                assert math.isclose(max(errors), r['max_abs'], rel_tol=2e-6, abs_tol=2e-7)
                cpu_refs += 1
                cpu_elements += 128
    assert c['identical_page_table_values_across_plans']
    assert len(c['timing_plan_orders']) == 7
    for trial, order in enumerate(c['timing_plan_orders']):
        expected_order = list(MODES[trial % len(MODES):] + MODES[:trial % len(MODES)])
        if trial % 2: expected_order.reverse()
        assert order == expected_order
    assert set(c['raw_ms']) == set(c['median_ms']) == set(MODES)
    for mode in MODES:
        raw = c['raw_ms'][mode]
        assert len(raw) == 7 and all(math.isfinite(v) and v > 0 for v in raw)
        assert statistics.median(raw) == c['median_ms'][mode]
        event_groups += 7
    timings[f'{n}/{context}'] = dict(median_ms=c['median_ms'],
        time_change_vs_current_K8V16_percent={m: 100 * (v / c['median_ms']['current'] - 1) for m, v in c['median_ms'].items()})
assert len(d['producer_records']) == 16 and gpu_refs == 32 and event_groups == 112
for r in d['producer_records'].values():
    plan = configuration(r['mode'], r['batch'], r['context'])
    assert r['IS_3D'] and r['key_dtype'] == 'torch.uint8' and r['value_dtype'] == 'torch.bfloat16'
    assert (r['tile'], r['warps'], r['segments'], r['stages']) == (plan['TILE_SIZE'], plan['num_warps'], plan['segments'], plan['num_stages'])
    assert r['n_regs'] is not None and r['n_spills'] is not None
from source import render
original_reader = artifact(proof['reader']['generated_source']).read_bytes()
text32, proof32 = render(original_reader)
address32 = d['address32_proof']
assert artifact(address32['generated_source']).read_text() == text32
assert all(address32[k] == v for k,v in proof32.items())
assert len(d['offset_bounds']) == 8 and all(0 <= x < 2**31 for bounds in d['offset_bounds'].values() for x in bounds)
result = dict(complete=True, only_integer_arithmetic_changed=True, all_16_changed_Graph_phases_bitwise_same=True, component_only=True, sources=len(sources), cases=8, plans=2,
    GPU_reference_records=gpu_refs, CPU_vectors_recomputed=cpu_refs, CPU_elements_recomputed=cpu_elements,
    event_groups=event_groups, timings=timings, adopted=False,
    full_engine_allocator_not_installed=True, prefill_not_tested=True, fresh_Math_not_run=True)
(root / 'local_download_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
