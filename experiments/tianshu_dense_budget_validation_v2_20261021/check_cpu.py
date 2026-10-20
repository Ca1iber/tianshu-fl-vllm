# SPDX-License-Identifier: Apache-2.0
import ast,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def main():
    import framework_profile
    assert framework_profile.parent.BASE==BASE.parent/"tianshu_dense_budget8192_v2_20261021" and framework_profile.runtime.BASE==BASE
    for path in BASE.glob('*.py'):ast.parse(path.read_text())
    spec=importlib.util.spec_from_file_location('_latest_budget_quality_composition',BASE.parent/'tianshu_dense_budget8192_v2_20261021/candidate_profile.py')
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    root=Path('/workspace/benchmark-results/20261021_dense_budget_v2_quality_CPU_gate_v2_tianshu');assert not root.exists();root.mkdir()
    command,env,profile,sources=parent.prepare(root,'candidate')
    cpu=json.loads((BASE.parent/'tianshu_dense_budget8192_v2_20261021/CPU_GATE_32g.json').read_text());expected=cpu['profiles'][1]
    assert sources==expected['frozen_sources'] and len(sources)==374 and len(profile)==64 and command==expected['command']
    prior='/workspace/benchmark-results/20261021_dense_budget8192_v2_CPU_profile_candidate_tianshu/'
    assert profile=={k:v.replace(prior,str(root)+'/') for k,v in expected['profile_environment'].items()}
    assert all(profile[k]=='candidate' for k in ('TIANSHU_DENSE_BUDGET_MODE','TIANSHU_LM_PRECISION_MODE','TIANSHU_PREFILL_LINEAR_MODE','TIANSHU_DENSE_INT8_MODE'))
    result=dict(complete=True,no_GPU_started=True,runtime_source_count=374,environment_count=64,
        exact_screened_runtime_composition=True,expected_budget=8192,expected_warm_points=99,
        actual_Math_C8_required=True,actual_official_C64_required=True,old_Math_not_reused=True,
        only_original_N256_4K_scheduled=True,sources=sources,controller_sources={str(p):parent.digest(p) for p in BASE.glob('*.py')})
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('sources','controller_sources')}))
if __name__=='__main__':main()
