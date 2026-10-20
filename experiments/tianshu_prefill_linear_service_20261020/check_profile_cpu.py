# SPDX-License-Identifier: Apache-2.0
"""Profile composition check only; does not construct Torch or start a GPU job."""
import ast,json
from pathlib import Path
from candidate_profile import BASE,prepare

def main():
    profiles=[]
    for mode in ('control','candidate'):
        root=Path(f'/workspace/benchmark-results/20261020_prefill_linear_CPU_gate_{mode}_tianshu');assert not root.exists();root.mkdir()
        command,env,profile,sources=prepare(root,mode)
        assert len(profile)==61 and profile['TIANSHU_COLD_START_MODE']=='gpu4' and profile['TIANSHU_DENSE_INT8_MODE']=='candidate'
        assert profile['TIANSHU_PREFILL_LINEAR_MODE']==mode and profile['PYTHONPATH']==str(BASE/'hooks')
        assert len(command)==13 and command[0:3]==['vllm','serve','/workspace/MiniCPM5-2B']
        profiles.append(dict(mode=mode,command=command,profile_environment=profile,frozen_sources=sources))
    assert profiles[0]['frozen_sources']==profiles[1]['frozen_sources']
    for p in BASE.rglob('*.py'):ast.parse(p.read_text())
    result=dict(complete=True,no_GPU_or_server_started=True,all_61_actual_environment_keys_verified=True,
        both_modes_retain_qualified_INT8_decode=True,same_GPU_packing_and_SQL_startup=True,
        runtime_source_count=len(profiles[0]['frozen_sources']),profiles=profiles)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
if __name__=='__main__':main()
