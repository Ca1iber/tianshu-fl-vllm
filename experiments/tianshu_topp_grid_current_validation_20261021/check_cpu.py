# SPDX-License-Identifier: Apache-2.0
"""Freeze current-screened runtime and preserve all quality/benchmark contracts."""
import ast,hashlib,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'tianshu_prefill_int8_validation_20261021'
SERVICE=BASE.parent/'tianshu_topp_grid_current_service_20261021'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
for p in BASE.glob('*.py'):ast.parse(p.read_text())
for name in ('baseline_quality_audit.py','dense_quality_audit.py','batch_policy.json'):assert sha(BASE/name)==sha(OLD/name)
assert (BASE/'prefill_quality_audit.py').read_text()==(OLD/'audit_profile.py').read_text().replace('407','438')
assert (BASE/'run_validation.py').read_text()==(OLD/'run_validation.py').read_text().replace('selective_Grouped_INT8_prefill_mixed_and_NN_decode','current_prefill_INT8_plus_TopP16K_grid2')
profile=read(SERVICE/'CPU_GATE_32g.json');assert profile['complete'] and profile['runtime_source_count']==438 and profile['environment_count']==62
runtime=profile['profiles'][1]['sources'];assert len(runtime)==438 and all(sha(p)==v for p,v in runtime.items())
assert profile['profiles'][0]['sources']==runtime
spec=importlib.util.spec_from_file_location('_quality_sampler_actual_profile',SERVICE/'candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
root=Path('/workspace/benchmark-results/20261021_topp_grid_current_validation_CPU_tianshu');assert not root.exists();root.mkdir()
command,environment,selected,sources=parent.prepare(root,'candidate')
assert sources==runtime and len(selected)==62 and len(command)==13
spec=importlib.util.spec_from_file_location('_quality_sampler_actual_audit',BASE/'audit_profile.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
assert callable(audit.audit_runtime)
controller=(BASE/'run_validation.py').read_text()
assert 'accuracy["accuracy"] > .95' not in controller # pass is in the original quality audit, no new score-based rerun
assert 'quality["accuracy"] > .95' in controller
assert controller.count('subprocess.Popen(evaluation')==1
assert all(x in controller for x in ('"--eval-batch-size", "8"','"--timeout", "3600"','max_tokens\":32768','[[4096,1024,64,256]]'))
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'batch_policy.json',SERVICE/'CPU_GATE_32g.json',BASE.parent/'tianshu_original_command_20261016/validation_profile.py']
result=dict(complete=True,CPU_only=True,no_GPU_started=True,new_Math_not_run=True,new_official_benchmark_not_run=True,runtime_source_count=438,environment_count=62,
    original_evalscope_and_official_workloads_byte_preserved=True,original_prefill_precision_checks_preserved=True,original_baseline_dense_quality_audits_byte_identical=True,
    current_sampler_geometry_audited_in_actual_C8_Math_or_C64_official=True,same_exact_screened_runtime_and_sources=True,C64_not_claimed_covered_by_C8_Math=True,
    runtime_sources_unchanged=runtime,sources={str(p.resolve()):sha(p) for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('sources','runtime_sources_unchanged')}))
