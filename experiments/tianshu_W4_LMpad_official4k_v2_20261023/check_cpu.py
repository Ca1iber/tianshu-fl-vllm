# SPDX-License-Identifier: Apache-2.0
import ast,json,sys
from pathlib import Path
from quality_profile import BASE,HOLDOUT,prepare,read,normalized_environment
for p in BASE.glob('*.py'):ast.parse(p.read_text())
root=Path('/workspace/benchmark-results/20261023_W4_LMpad_v2_official4k_CPU_tianshu');assert not root.exists() and not (BASE/'CPU_GATE_32g.json').exists();root.mkdir();command,env,profile,sources=prepare(root,HOLDOUT,batch=8,scope='prefill',check_screen=False,check_quality=False)
mathroot=Path('/workspace/benchmark-results/20261023_W4_LMpad_v2_math8_tianshu');current=read(mathroot/'state.json');assert sources==current['frozen_sources'] and command==current['server_command'] and normalized_environment(profile,root)==normalized_environment(current['profile_environment'],mathroot)
assert len(sources)==944 and len(command)==13 and len(profile)==62 and len(profile|{'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']})==63 and not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
g=dict(complete=True,CPU_only=True,model_not_started=True,official_screen_pending=True,full_quality_pending=True,GPU_runtime_requires_full_quality_and_positive_short=True,source_count=944,environment_count=63,original13argv=True,controller_sources=read(root/'controller_sources.json'))
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps({k:v for k,v in g.items() if k!='controller_sources'}))
