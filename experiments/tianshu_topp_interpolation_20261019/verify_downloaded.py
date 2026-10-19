# SPDX-License-Identifier: Apache-2.0
"""Recompute strict per-row semantic gates and timing from complete TopP raw records."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from router import PLANS
from precision import render_source, topk_ast


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic(record, batch):
    keys = ("kept_mass", "smallest_kept_probability", "minimum_kept_logit", "maximum_removed_logit", "nonempty", "CPU_p")
    assert all(len(record[key]) == batch for key in keys)
    rows = zip(*(record[key] for key in keys))
    strict = all(nonempty and mass >= p - 1e-6 and mass - smallest <= p + 1e-6 and removed <= kept
                 for mass, smallest, kept, removed, nonempty, p in rows)
    assert record["CPU_strict_semantics"] == strict
    assert all(abs(p - .95) < 1e-6 for p in record["CPU_p"])
    assert record["relative_gate"] == (record["retained_values_unchanged"] and
        (strict if record["strict_required"] else (record["exact_adopted_baseline"] or strict)))
    if record["exact_adopted_baseline"]:
        assert record["adopted_mask_difference"] == 0
    return strict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    state = json.loads((root.parent / "result.json").read_text())
    data = json.loads((root / "result.json").read_text())
    assert state["phase"] == "complete" and data["complete"] and data["numerical_checks_passed"]
    assert not data["official_benchmark"] and not data["adopted"]
    assert data["plans"] == [list(plan) for plan in PLANS] and len(data["cases"]) == 10
    assert data["CPU_tolerance"] == 1e-6 and data["baseline"] == "native8192/4096/w16/s1"
    assert data["timing_includes_identical_input_restore"]
    assert data["RNG_implementation_and_sampling_parameters_unchanged"] and data["sampling_RNG_not_exercised"]
    assert data["current_best_controls"] == ["fixed32_geo", "fixed16_geo"]
    base = Path(__file__).resolve().parent
    for source in (base / "probe_precision.py", base / "router.py"):
        assert source.read_bytes() == (root / "source_snapshot" / source.name).read_bytes()
    assert sha(root / "source_snapshot/native_topk_topp_triton.py") == data["native_source_sha256"] == "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"
    assert data["corrected_candidates_require_strict_CPU"]
    native_source=root/"source_snapshot/native_topk_topp_triton.py"
    assert data['precision_generation']['base_geo_sha256']=='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
    assert data['precision_generation']['strict_CPU_tolerance_unchanged']==1e-6
    assert data['precision_generation']['p_unchanged'] and data['precision_generation']['RNG_implementation_unchanged']
    for mode,record in data['precision_generation']['modes'].items():
        generated=root/'precision_source'/f'generated_{mode}.py'
        assert generated.read_text()==render_source(native_source.read_bytes(),mode)
        assert sha(generated)==record['sha256']
        assert topk_ast(generated.read_text())==topk_ast(native_source.read_text())
    assert set(data['precision_generation']['modes']) == {'geo', 'secant', 'illinois'}
    assert data['precision_generation']['standalone_search_limit_unchanged'] == 32
    assert data['precision_generation']['standalone_interval_tolerance_unchanged'] == 1e-12
    assert data['precision_generation']['all_GPU_arithmetic_FP32']
    assert (base/'precision.py').read_bytes()==(root/'source_snapshot/precision.py').read_bytes()
    assert len(state['frozen_sources']) == 12
    repo=base.parents[1]
    for source,digest in state['frozen_sources'].items():
        assert sha(root.parent/'source_snapshot'/Path(source).relative_to('/'))==digest
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
            assert sha(repo/Path(source).relative_to('/workspace/vllm-plugin-FL'))==digest
    cases = [(8, "normal", False), (32, "normal", False), (64, "normal", False),
             (64, "peaked", False), (8, "masked", False), (8, "ties", False), (32, "normal", True),
             (64,"bf16_normal",False),(64,"bf16_peaked",False),(8,"all_equal",False)]
    passing = {plan[0]: True for plan in PLANS}
    events, gate_records, errors, rows = 0, 0, [], []
    for case, (batch, distribution, topk) in zip(data["cases"], cases):
        assert (case["batch"], case["distribution"], case["topk"]) == (batch, distribution, topk)
        assert len(case["variants"]) == len(PLANS)
        baseline = case["variants"][0]
        assert baseline["label"] == "adopted" and "median_ms" in baseline
        baseline_grid = baseline["compiled_metadata"]["grid"]
        assert len(baseline_grid) == 1 and 1 <= baseline_grid[0] <= batch
        for variant, plan in zip(case["variants"], PLANS):
            name, block, trunc, warps = plan
            assert variant["label"] == name and variant["requested_plan"] == list(plan)
            if "error" in variant:
                assert name != "adopted" and "median_ms" not in variant
                errors.append(dict(batch=batch, distribution=distribution, label=name, error=variant["error"]))
                passing[name] = False
                continue
            metadata = variant["compiled_metadata"]
            fixed=name.startswith("fixed")
            active=fixed and not topk
            if fixed and topk:block,trunc,warps=8192,4096,16
            assert metadata["corrected_source_active"]==active
            mode=name.rsplit('_', 1)[-1] if active else 'geo'
            assert metadata['source_module']==(f'_tianshu_interpolated_topp_{mode}' if active else 'vllm.v1.sample.ops.topk_topp_triton')
            assert metadata["label"] == name and metadata["BLOCK_SIZE"] == block and metadata["BLOCK_SIZE_TRUNC"] == trunc
            assert metadata["num_warps"] == warps and metadata["num_stages"] == 1
            assert metadata["VOCAB_SIZE"] == 130560 and metadata["TOPK_ENABLED"] == topk and metadata["TOPP_ENABLED"]
            assert metadata["grid"] == baseline_grid
            assert variant["original"]["strict_required"]==fixed
            semantic(variant["original"], batch)
            gate_records += batch
            if "changed" in variant:
                assert variant["changed"]["strict_required"]==fixed
                semantic(variant["changed"], batch)
                gate_records += batch
            valid = bool(variant["original"]["relative_gate"] and variant.get("changed", {}).get("relative_gate")
                         and variant.get("original_graph_exact") and variant.get("changed_graph_exact"))
            assert ("median_ms" in variant) == valid
            passing[name] &= valid
            if valid:
                raw = variant["raw_ms"]
                assert len(raw) == 7 and all(t > 0 for t in raw) and statistics.median(raw) == variant["median_ms"]
                events += len(raw)
                rows.append(dict(batch=batch, distribution=distribution, topk=topk, label=name,
                    baseline_ms=baseline["median_ms"], candidate_ms=variant["median_ms"],
                    time_change_percent=100 * (variant["median_ms"] / baseline["median_ms"] - 1)))
    assert data["relative_passing_plans"] == [plan[0] for plan in PLANS if passing[plan[0]]]
    assert passing["adopted"] and passing["identity"]
    artifacts = list((root / "kernel_artifacts").glob("*/manifest.json"))
    assert {p.parent.name for p in artifacts} >= {"adopted", "identity"}
    for path in artifacts:
        record = json.loads(path.read_text())
        assert record["metadata"]["label"] == path.parent.name and record["files"]
        for name, digest in record["files"].items(): assert sha(path.parent / name) == digest
    result = dict(complete=True, official_benchmark=False, adopted=False, cases=10,
                  requested_configurations=len(PLANS), per_row_semantic_gate_records_recomputed=gate_records,
                  raw_events_recomputed=events, relative_passing_plans=data["relative_passing_plans"],
                  baseline_strict_errors_retained=True, changed_input_graphs_verified=True,
                  source12_native_and_three_generated_bytes_verified=True, TopK_AST_and_actual_native_fallback_verified=True, errors=errors, rows=rows)
    comparisons = []
    for case in data['cases']:
        lookup = {v['label']: v for v in case['variants']}
        for candidate, control in [('fixed32_secant', 'fixed32_geo'),
                ('fixed32_illinois', 'fixed32_geo'), ('fixed16_illinois', 'fixed16_geo')]:
            c, b = lookup[candidate], lookup[control]
            if 'median_ms' in c and 'median_ms' in b:
                comparisons.append(dict(batch=case['batch'], distribution=case['distribution'],
                    topk=case['topk'], candidate=candidate, control=control,
                    candidate_ms=c['median_ms'], control_ms=b['median_ms'],
                    time_change_percent=100*(c['median_ms']/b['median_ms']-1)))
    result['current_best_control_comparisons'] = comparisons
    result['sampling_RNG_not_exercised'] = True
    result['service_not_tested'] = True
    (root / "local_download_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "current_best_control_comparisons")}))


if __name__ == "__main__": main()
