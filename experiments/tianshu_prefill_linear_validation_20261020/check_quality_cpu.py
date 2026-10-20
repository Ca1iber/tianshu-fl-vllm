# SPDX-License-Identifier: Apache-2.0
import ast,importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent

def main():
    for path in BASE.glob('*.py'):ast.parse(path.read_text())
    spec=importlib.util.spec_from_file_location('prefill_quality_CPU_composition',BASE.parent/'tianshu_prefill_linear_service_20261020/candidate_profile.py')
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    root=Path('/workspace/benchmark-results/20261020_prefill_linear_quality_CPU_gate_tianshu');assert not root.exists();root.mkdir()
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(sources)==345 and len(profile)==61 and len(command)==13
    expected=json.loads((BASE.parent/'tianshu_prefill_linear_service_20261020/CPU_GATE_32g.json').read_text())['profiles'][1]
    assert sources==expected['frozen_sources'] and command==expected['command']
    assert profile=={k:v.replace('/workspace/benchmark-results/20261020_prefill_linear_v2_CPU_gate_candidate_tianshu/',str(root)+'/') for k,v in expected['profile_environment'].items()}
    result=dict(complete=True,no_GPU_or_server_started=True,runtime_source_count=345,environment_count=61,exact_service_candidate_profile=True,
        fresh_105_Math_required_before_N256=True,no_regular_16K=True)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
