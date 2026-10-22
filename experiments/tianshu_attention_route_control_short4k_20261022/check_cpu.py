# SPDX-License-Identifier: Apache-2.0
import ast,json,sys
from pathlib import Path
import candidate_profile as p
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261022_attention_route_control_short4k_CPU_tianshu');assert not ROOT.exists();ROOT.mkdir()
for path in BASE.glob('*.py'):ast.parse(path.read_text())
command,environment,profile,sources=p.prepare(ROOT,'control','prefill');p.frozen_check(sources);p.frozen_check(p.read(ROOT/'controller_sources.json'));assert len(sources)==903 and len(command)==13 and len({**profile,'TRITON_CACHE_DIR':environment['TRITON_CACHE_DIR']})==63 and profile['PYTHONPATH'].endswith('/hooks_control');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
gate=dict(complete=True,CPU_only=True,model_not_started=True,model_sources=903,environment_count=63,original13_argv=True,mode='same_hook_BF16_control',fresh_quality_not_claimed_for_this_control=True,controllers=p.read(ROOT/'controller_sources.json'));(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='controllers'}))
