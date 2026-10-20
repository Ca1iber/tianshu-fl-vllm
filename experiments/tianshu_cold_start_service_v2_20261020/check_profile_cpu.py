# SPDX-License-Identifier: Apache-2.0
"""Validate three complete loader profiles without constructing Torch."""
import ast,json
from pathlib import Path
from candidate_profile import BASE,prepare

def main():
    rows=[]
    for mode in ('native','cpu4','gpu4'):
        root=Path(f'/workspace/benchmark-results/20261020_cold_start_v2_CPU_gate_{mode}_tianshu');assert not root.exists();root.mkdir()
        command,env,profile,sources=prepare(root,mode)
        assert len(profile)==59 and profile['TIANSHU_COLD_START_MODE']==mode
        assert profile['TIANSHU_SQL_REFLECTION_PROBE']=='1' and profile['TIANSHU_DENSE_INT8_MODE']=='candidate'
        assert len(command)==13
        rows.append(dict(mode=mode,command=command,profile_environment=profile,frozen_sources=sources))
    assert len({json.dumps(r['frozen_sources'],sort_keys=True) for r in rows})==1
    for p in BASE.rglob('*.py'):ast.parse(p.read_text())
    result=dict(complete=True,no_GPU_or_server_started=True,environment_count=59,runtime_source_count=len(rows[0]['frozen_sources']),
        same_already_adopted_SQL_reflect_one_in_all_modes=True,original_W8_active_in_all_modes=True,profiles=rows)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
if __name__=='__main__':main()
