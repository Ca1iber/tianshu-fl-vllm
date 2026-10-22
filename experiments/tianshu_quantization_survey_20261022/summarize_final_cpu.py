# SPDX-License-Identifier: Apache-2.0
"""Audit completed quantization evidence and compare measured component timings."""
import csv
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
MODEL = Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
MODEL_SHA = '14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
EXPERIMENTS = [
    'tianshu_quant_weight_component_v2_20261022',
    'tianshu_quant_weight_pairload_20261022',
    'tianshu_w8a16_postscale_20261022',
    'tianshu_w4a16_group_postscale_20261022',
    'tianshu_w8a16_nt_postscale_20261022',
    'tianshu_w4a16_nt_group_postscale_20261022',
    'tianshu_w8a16_native_tiles_20261022',
    'tianshu_w4a8_group_nt_20261022',
    'tianshu_w4a8_group_nn_stages_20261022',
]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def audit_evidence(folder, weight_component):
    directory = ROOT / folder
    cpu_path = directory / 'INDEPENDENT_CPU_GATE_32g.json'
    gpu_path = directory / 'GPU_GATE_32g.json'
    csv_path = directory / 'KEY_TIMINGS_32g.csv'
    cpu, gpu = load(cpu_path), load(gpu_path)
    assert cpu['complete'] and cpu['CPU_only'] and gpu['complete']
    assert cpu['raw_result_sha256'] == sha(gpu_path)
    assert cpu['raw_result_sha256'] == sha(Path(cpu['raw_directory']) / 'result.json')
    assert cpu['key_timings_sha256'] == sha(csv_path)
    sources = gpu['frozen_sources'] if weight_component else gpu['sources']
    assert sources and all(sha(path) == digest for path, digest in sources.items())
    assert gpu['adopted'] is False and gpu['model_executed'] is False
    if weight_component:
        assert gpu['official_benchmark'] is False and cpu['sources_unchanged']
        assert gpu['checkpoint_sha256_before'] == gpu['checkpoint_sha256_after'] == MODEL_SHA
        assert gpu['checkpoint_unchanged'] and cpu['checkpoint_unchanged']
        assert all(c['complete'] and not c.get('compile_failed', False) for c in gpu['cases'])
        assert cpu['phase_artifacts_verified'] == 3 * len(gpu['cases'])
    rows = list(csv.DictReader(csv_path.open()))
    if weight_component:
        assert len(rows) == len(gpu['cases'])
    record = dict(experiment=folder, CPU_gate_sha256=sha(cpu_path),
                  GPU_gate_sha256=sha(gpu_path), key_timings_sha256=sha(csv_path),
                  frozen_source_files=len(sources))
    if weight_component:
        record.update(cases=len(gpu['cases']), phase_artifacts=cpu['phase_artifacts_verified'],
                      selected_FP64_outputs=cpu['independent_selected_FP64_outputs'],
                      selected_W4_INT32_partials_exact=cpu.get('selected_W4_INT32_partials_exact', 0))
    return record, rows

def main():
    evidence, comparison = [], []
    for folder in EXPERIMENTS:
        record, rows = audit_evidence(folder, True)
        evidence.append(record)
        for role, m in [('down', 64), ('lm_head', 8), ('lm_head', 64)]:
            selected = [r for r in rows if (r['role'], int(r['M'])) == (role, m)]
            if not selected:
                continue
            best = min(selected, key=lambda r: float(r['candidate_ms']))
            control = 'W8A8' if role == 'lm_head' and m == 64 else 'BF16'
            candidate_ms, control_ms = float(best['candidate_ms']), float(best[f'{control}_ms'])
            assert candidate_ms > control_ms
            comparison.append(dict(experiment=folder, role=role, M=m, format=best['format'],
                                   tile_id=int(best['tile_id']), candidate_ms=candidate_ms,
                                   adopted_control=control, control_ms=control_ms,
                                   candidate_duration_ratio=candidate_ms/control_ms))
    a8_record, a8_rows = audit_evidence('tianshu_a8_reciprocal_20261022', False)
    assert len(a8_rows) == 6
    assert all(float(r['half_div_us']) > float(r['old_us']) and
               float(r['half_fma_us']) > float(r['old_us']) for r in a8_rows)
    model_sha = sha(MODEL)
    assert model_sha == MODEL_SHA
    summary_path = BASE / 'FINAL_COMPONENT_COMPARISON_32g.csv'
    with summary_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(comparison)
    gate = dict(complete=True, CPU_only=True, component_versions=len(evidence),
                total_component_cases=sum(r['cases'] for r in evidence),
                total_phase_artifacts=sum(r['phase_artifacts'] for r in evidence),
                selected_FP64_outputs=sum(r['selected_FP64_outputs'] for r in evidence),
                selected_W4_INT32_partials_exact=sum(r['selected_W4_INT32_partials_exact'] for r in evidence),
                every_listed_target_slower_than_adopted_control=True,
                components=evidence, A8_reciprocal=a8_record,
                A8_corrected_variants_slower_at_all_six_tested_points=True,
                original_checkpoint_sha256=model_sha, original_checkpoint_bytes=MODEL.stat().st_size,
                model_executed=False, new_model_quality_proven=False, adopted=False,
                official_benchmark_executed=False, summary_CSV_sha256=sha(summary_path),
                summary_source_sha256=sha(__file__))
    (BASE / 'FINAL_AUDIT_GATE_32g.json').write_text(json.dumps(gate, indent=2) + '\n')
    print(json.dumps({key: value for key, value in gate.items() if key not in ('components', 'A8_reciprocal')}))

if __name__ == '__main__':
    main()
