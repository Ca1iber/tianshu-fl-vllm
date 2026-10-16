# SPDX-License-Identifier: Apache-2.0
"""Audit static INT8 fixtures and folded-scale component records."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
from fold_value_scale import render_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    state = json.loads((root / 'result.json').read_text())
    assert state['phase'] == 'complete' and state['complete']
    assert not state['official_benchmark'] and not state['adopted']
    assert state['manual_quantization_not_native_writer'] and state['full_model_and_Math_not_run']
    repo = Path(__file__).resolve().parents[2]
    for name, expected in state['source_snapshots'].items():
        assert digest(root / 'source_snapshot' / Path(name).relative_to('/')) == expected, name
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert digest(repo / Path(name).relative_to('/workspace/vllm-plugin-FL')) == expected, name
    original = next(root.glob('decode_coherent_records/cache_generated/*/generated_cache_policy.py'))
    generated = root / 'value_scale_generated/generated_value_scale.py'
    text, record = render_source(original.read_bytes())
    assert generated.read_text() == text
    for key, value in record.items():
        assert state['derivation'][key] == value
    def producer(tree):
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_unified_attention')
    assert ast.dump(producer(ast.parse(original.read_text()))) == ast.dump(producer(ast.parse(text)))
    variants = {'BF16', 'INT8_native_tensor', 'INT8_folded_w1', 'INT8_folded_w2'}
    assert len(state['cases']) == 3
    assert {(c['batch'], c['context']) for c in state['cases']} == {(8, 33), (64, 4096), (64, 5120)}
    timings, errors, events = [], [], 0
    for case in state['cases']:
        n = case['batch']
        assert len(case['lengths']) == n and max(case['lengths']) == case['context']
        assert len(case['quantization']) == 2
        for q in case['quantization']:
            assert q['K_scale'] == q['V_scale'] == 1 / 16
            assert q['elements_per_K_or_V'] > 0
            assert 0 <= q['K_clipped'] <= q['elements_per_K_or_V']
            assert 0 <= q['V_clipped'] <= q['elements_per_K_or_V']
        results = case['results']
        assert len(results) == 4 and {r['name'] for r in results} == variants
        baseline = next(r for r in results if r['name'] == 'BF16')['median_ms']
        for r in results:
            assert len(r['raw_ms']) == 7 and all(t > 0 for t in r['raw_ms'])
            assert statistics.median(r['raw_ms']) == r['median_ms']
            assert abs(100 * (r['median_ms'] / baseline - 1) - r['time_change_percent']) < 1e-12
            events += 7
            m = r['metadata']
            assert m['grid'] == [n // 2 + n, 2, 8 if n == 64 else 16]
            assert m['quant_mode'] == (1 if r['name'] == 'INT8_native_tensor' else 0)
            assert m['key_dtype'] == ('torch.bfloat16' if r['name'] == 'BF16' else 'torch.int8')
            assert m['compiled_warps'] == m['warps'] == (2 if r['name'].endswith('w2') else 1)
            assert m['tile'] == (32 if r['name'].endswith('w2') else 16)
            assert r['key_strides'] == [4096, 256, 128, 1]
            assert r['scale_folded'] == ('folded' in r['name'])
            timings.append(dict(batch=n, context=case['context'], **r))
        assert len(case['phases']) == 2 and [p['phase'] for p in case['phases']] == [0, 1]
        for p in case['phases']:
            assert p['outputs_cleared_before_replay']
            for key in ('CPU_BF16_reference', 'CPU_dequantized_reference'):
                assert p[key] and all(c['passed'] and c['failed_elements'] == 0 for c in p[key])
            assert len(p['checks']) == 4 and {c['name'] for c in p['checks']} == variants
            for c in p['checks']:
                check = c['against_own_FP32']
                assert check['passed'] and check['failed_elements'] == 0
                assert check['atol'] == .02 and check['rtol'] == .01 and c['tails_preserved']
                assert all(math.isfinite(v) and v >= 0 for v in c['against_BF16_FP32'].values())
                errors.append(dict(batch=n, context=case['context'], phase=p['phase'], name=c['name'], **c['against_BF16_FP32']))
    audit = dict(complete=True, source_snapshots_verified=len(state['source_snapshots']),
                 producer_AST_unchanged=True, generated_reducer_reconstructed=True,
                 all_raw_events_recomputed=events, six_changed_input_Graph_phase_records_checked=True,
                 full_GPU_tensor_values_not_serialized=True, no_native_writer_service_or_Math=True,
                 INT8_storage_not_FP8=True, adopted=False, timings=timings, approximations=errors)
    (root / 'local_download_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k not in ('timings', 'approximations')}))


if __name__ == '__main__':
    main()
