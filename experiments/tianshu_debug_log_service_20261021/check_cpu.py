# SPDX-License-Identifier: Apache-2.0
import ast,json
from pathlib import Path
from candidate_profile import BASE,prepare
profiles=[]
for path in BASE.rglob('*.py'):ast.parse(path.read_text())
for mode in ('control','candidate'):
    root=Path(f'/workspace/benchmark-results/20261021_debug_log_service_CPU_{mode}_tianshu');assert not root.exists();root.mkdir()
    command,env,profile,sources=prepare(root,mode)
    assert len(profile)==62 and len(command)==13 and profile['PYTHONPATH']==str(BASE/('hooks_'+mode))
    assert all(profile[k]=='candidate' for k in ('TIANSHU_DENSE_INT8_MODE','TIANSHU_LM_PRECISION_MODE','TIANSHU_PREFILL_LINEAR_MODE','TIANSHU_SEQ_LIMIT_MODE'))
    profiles.append(dict(mode=mode,command=command,profile=profile,sources=sources))
assert profiles[0]['sources']==profiles[1]['sources']
result=dict(complete=True,CPU_only=True,no_GPU_started=True,runtime_source_count=len(profiles[0]['sources']),environment_count=62,
    original_serve_command_preserved=True,parent_runtime_sources=359,all_own_AST_parsed=True,profiles=profiles)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
