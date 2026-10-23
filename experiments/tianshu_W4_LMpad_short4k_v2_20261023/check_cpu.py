# SPDX-License-Identifier: Apache-2.0
import ast,json,sys
from pathlib import Path
from candidate_profile import BASE,EXP,prepare,read,digest,normalized_environment
root=Path('/workspace/benchmark-results/20261023_W4_LMpad_short4k_v2_CPU_tianshu');assert not root.exists() and not (BASE/'CPU_GATE_32g.json').exists();root.mkdir();arms={}
for p in BASE.glob('*.py'):ast.parse(p.read_text())
for name in ('control','candidate'):
 arm=root/name;arm.mkdir();command,env,profile,sources=prepare(arm,name,check_quality=False);assert len(command)==13 and len(profile|{'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']})==63;arms[name]=dict(command=command,profile=normalized_environment(profile,arm),sources=sources)
a,b=arms['control'],arms['candidate'];assert a['command']==b['command'] and len(a['sources'])==903 and len(b['sources'])==944 and all(b['sources'][p]==h for p,h in a['sources'].items());assert {k for k in a['profile'] if a['profile'][k]!=b['profile'][k]}=={'PYTHONPATH'}
current=Path('/workspace/benchmark-results/20261023_W4_LMpad_v2_math8_tianshu');s=read(current/'state.json');observed=read(current/'observed_full_server_environment.json');assert s['frozen_sources']==b['sources'] and observed['command']==b['command'] and len(observed['environment'])==63
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
g=dict(complete=True,CPU_only=True,model_not_started=True,baseline_model_sources=903,candidate_model_sources=944,actual_current_C8_13argv63env_model_SHA_verified=True,only_environment_difference='PYTHONPATH',fresh_full_quality_pending=True,GPU_runtime_requires_full_independent_C8_C64_quality=True,new_model_sources_unchanged=True)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps(g))
