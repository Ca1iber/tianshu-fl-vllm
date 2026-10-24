# SPDX-License-Identifier: Apache-2.0
"""Freeze the exact new composition without borrowing previous model quality."""
import ast,json,os
from pathlib import Path
from candidate_profile import BASE,BM8,prepare,digest,read,frozen_check,released_sources

def main():
    root=Path('/dev/shm/tianshu_r5_BM8_CPU_20261024')
    assert not root.exists();root.mkdir()
    command,env,profile,sources=prepare(root,'candidate',check_gate=False)
    assert len(command)==13 and len(profile)+1==67
    assert profile['TIANSHU_QKVO64_FITTED_MODE']==profile['TIANSHU_R5_BM8_MODE']=='candidate'
    for p in BASE.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    released=released_sources();assert all(sources[p]==h for p,h in released.items())
    selection=read(BM8/'selected_plan.json');gate=read(selection['component_gate'])
    frozen_check(sources)
    result=dict(complete=True,model_ready=True,CPU_only=True,sources=sources,
        original997_exact_subset=True,original997_source_count=len(released),source_count_without_self=len(sources),
        parent_prepare_always_candidate=True,parent_QKVO_fitted_in_both_arms=True,
        source_snapshot_directory=str(root/'source_snapshot'),immutable_SDK_snapshots_reused=True,
        selected_component_source=selection['kernel_source'],selected_component_sha256=selection['kernel_sha256'],
        original_component_full_bitwise_elements=gate['full_output_elements_rechecked'],
        old_component_qualification_reused_but_no_old_model_scores=True,
        fresh_C8_C64_quality_required=True,service_and_performance_not_yet_run=True,
        controller=str(BASE/'run_model.py'),controller_sha256=digest(BASE/'run_model.py'),
        candidate_profile_environment=profile,original_serve_argv=command)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','model_ready','source_count_without_self','original997_source_count')}),flush=True)
if __name__=='__main__':main()
