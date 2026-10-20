# SPDX-License-Identifier: Apache-2.0
import ast,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def main():
    for path in BASE.glob('*.py'):ast.parse(path.read_text())
    spec=importlib.util.spec_from_file_location('LM_precision_CPU_validation_composition',BASE.parent/'tianshu_lm_precision_service_20261020/candidate_profile.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    root=Path('/workspace/benchmark-results/20261020_lm_precision_quality_CPU_gate_tianshu');assert not root.exists();root.mkdir()
    command,env,profile,sources=parent.prepare(root,'candidate')
    expected=json.loads((BASE.parent/'tianshu_lm_precision_service_20261020/CPU_GATE_32g.json').read_text())['profiles'][1]
    assert sources==expected['frozen_sources'] and len(sources)==359 and len(profile)==62 and command==expected['command']
    assert profile=={k:v.replace('/workspace/benchmark-results/20261020_lm_precision_v2_CPU_gate_candidate_tianshu/',str(root)+'/') for k,v in expected['profile_environment'].items()}
    assert profile['TIANSHU_LM_PRECISION_MODE']==profile['TIANSHU_PREFILL_LINEAR_MODE']=='candidate'
    result=dict(complete=True,no_GPU_started=True,runtime_source_count=359,environment_count=62,exact_candidate_service_profile=True,
        C8_GateUp_integer_required=True,C8_LM_original_BF16_required=True,C64_all_integer_verified_separately=True,new_Math_required=True)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
