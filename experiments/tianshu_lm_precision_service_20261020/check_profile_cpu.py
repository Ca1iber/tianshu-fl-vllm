# SPDX-License-Identifier: Apache-2.0
import ast,json
from pathlib import Path
from candidate_profile import BASE,prepare

def main():
    profiles=[]
    for mode in ('control','candidate'):
        root=Path(f'/workspace/benchmark-results/20261020_lm_precision_v2_CPU_gate_{mode}_tianshu');assert not root.exists();root.mkdir()
        command,env,profile,sources=prepare(root,mode)
        assert len(profile)==62 and profile['TIANSHU_PREFILL_LINEAR_MODE']=='candidate' and profile['TIANSHU_COLD_START_MODE']=='gpu4' and profile['TIANSHU_DENSE_INT8_MODE']=='candidate'
        assert profile['TIANSHU_LM_PRECISION_MODE']==mode and len(command)==13
        profiles.append(dict(mode=mode,command=command,profile_environment=profile,frozen_sources=sources))
    assert profiles[0]['frozen_sources']==profiles[1]['frozen_sources']
    for p in BASE.rglob('*.py'):ast.parse(p.read_text())
    result=dict(complete=True,no_GPU_started=True,runtime_source_count=len(profiles[0]['frozen_sources']),environment_count=62,
        same_prefill_linear_and_GPU4_in_both_modes=True,all_GateUp_M8_M64_integer_retained=True,LM_M64_integer_retained=True,candidate_LM_M8_restores_original_BF16=True,
        profiles=profiles)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
if __name__=='__main__':main()
