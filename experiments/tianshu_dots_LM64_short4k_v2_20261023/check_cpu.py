# SPDX-License-Identifier: Apache-2.0
"""Reuse exact already-materialized model snapshots; only the quality gate changed."""
import ast,json,sys
from pathlib import Path
from candidate_profile import BASE,EXP,read,digest,frozen_check,normalized_environment
for p in BASE.glob('*.py'):ast.parse(p.read_text())
published=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');model_dir=EXP/'tianshu_dotsaddz_LM64_service_20261023';model=read(model_dir/'CPU_GATE_32g.json');candidate={**model['sources'],str(model_dir/'CPU_GATE_32g.json'):digest(model_dir/'CPU_GATE_32g.json')};control=published['frozen_sources'];assert len(control)==903 and len(candidate)==934 and all(candidate[p]==h for p,h in control.items())
previous=Path('/workspace/benchmark-results/20261023_dots_LM64_short4k_CPU_tianshu')
for arm,sources in [('control',control),('candidate',candidate)]:
 frozen_check(sources)
 for p,h in sources.items():assert digest(previous/arm/'source_snapshot'/Path(p).relative_to('/'))==h
current=Path('/workspace/benchmark-results/20261023_dots_LM64_math8_tianshu');observed=read(current/'observed_full_server_environment.json');state=read(current/'state.json');assert state['frozen_sources']==candidate and observed['command']==state['server_command']==published['server_command'] and len(observed['command'])==13 and len(observed['environment'])==63 and observed['environment']==state['profile_environment']|state['additional_environment']
a=normalized_environment(published['environment'],published['old_output_root']);b=normalized_environment(observed['environment'],current);assert {k for k in a if a[k]!=b[k]}=={'PYTHONPATH'}
recovery=read(EXP/'tianshu_dots_LM64_quality_results_v2_20261023/C64_QUALITY_GATE_32g.json');assert recovery['complete'] and not recovery['model_quality_qualified'] and recovery['C64_correct']==101 and recovery['C64_original_audit_failed_reconciled_CPU'] and not recovery['C64_GPU_rerun'];frozen_check(recovery['CPU_reconciliation_sources'])
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
gate=dict(complete=True,CPU_only=True,model_not_started=True,prior_exact_model_CPU_snapshots_reused_and_all_SHA_verified=True,actual_current934_C8_13argv63env_verified=True,baseline_model_sources=903,candidate_model_sources=934,only_environment_difference='PYTHONPATH',C64_CPU_reconciliation_explicitly_checked=True,fresh_full_quality_pending=True,runtime_requires_full_independent_C8_C64_quality=True,new_numerical_or_model_source_changes=False)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
