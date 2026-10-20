# SPDX-License-Identifier: Apache-2.0
"""Controller-source scope and exact runtime freeze proof; no new GPU or Math."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'tianshu_lm_precision_validation_20261020'
SERVICE=BASE.parent/'tianshu_prefill_int8_service_20261021'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected=(OLD/'audit_profile.py').read_text()
changes=[('    assert len({r["prefix"] for r in actual})==42 and all(r["implementation"]=="linear_contiguous" and r["calls"]>0 for r in actual)','    assert all(r["implementation"]=="linear_contiguous" and r["calls"]>0 for r in actual)'),
    ('actual_new_prefill_path_all_42_GateUp_verified=True','actual_BF16_fallback_prefill_prefixes=len({r["prefix"] for r in actual})')]
for before,after in changes:assert expected.count(before)==1;expected=expected.replace(before,after)
assert expected==(BASE/'dense_quality_audit.py').read_text()
assert sha(BASE/'baseline_quality_audit.py')==sha(OLD/'baseline_quality_audit.py') and sha(BASE/'batch_policy.json')==sha(OLD/'batch_policy.json')
assert (BASE/'run_validation.py').read_text()==(OLD/'run_validation.py').read_text().replace('W8A8_GateUp_LM64_BF16_LM_small_and_prefill_linear','selective_Grouped_INT8_prefill_mixed_and_NN_decode')
for p in BASE.glob('*.py'):ast.parse(p.read_text())
profile=json.loads((SERVICE/'CPU_GATE_32g.json').read_text());assert profile['complete'] and profile['runtime_source_count']==407 and profile['environment_count']==62
runtime=profile['profiles'][1]['sources'];assert all(sha(p)==v for p,v in runtime.items())
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'batch_policy.json',SERVICE/'CPU_GATE_32g.json',BASE.parent/'tianshu_original_command_20261016/validation_profile.py']
result=dict(complete=True,CPU_only=True,no_GPU_started=True,new_Math_not_run=True,new_official_benchmark_not_run=True,runtime_source_count=407,environment_count=62,
    exactly_two_BF16_coverage_quality_audit_adaptations=True,other_dense_quality_audit_text_preserved=True,baseline_quality_audit_and_policy_byte_identical=True,
    original_evaluation_and_official_workload_controller_preserved=True,all_AST_parsed=True,runtime_sources_unchanged=runtime,sources={str(p.resolve()):sha(p) for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('sources','runtime_sources_unchanged')}))
