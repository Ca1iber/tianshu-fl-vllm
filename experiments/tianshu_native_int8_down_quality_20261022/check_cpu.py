# SPDX-License-Identifier: Apache-2.0
import ast,json,sys
from pathlib import Path
import quality_profile as q
BASE=Path(__file__).resolve().parent;root=Path('/workspace/benchmark-results/20261022_native_int8_down_quality_CPU_tianshu');assert not root.exists();root.mkdir()
for path in BASE.glob('*.py'):ast.parse(path.read_text())
original=q.read(BASE.parent/'tianshu_native_int8_prefill_release_v2_20261022/serve_profile_template.json')['frozen_sources'];sources0=None;checks=[]
for scope,batch in [('all',64),('all',8),('prefill',64),('prefill',8)]:
 out=root/f'{scope}_C{batch}';out.mkdir();command,environment,profile,sources=q.prepare(out,q.HOLDOUT,batch=batch,scope=scope);assert len(command)==13 and len(profile)==62 and all(sources[p]==h for p,h in original.items());q.frozen_check(sources);controllers=q.read(out/'controller_sources.json');q.frozen_check(controllers)
 if sources0 is None:sources0=sources
 else:assert sources==sources0
 full={**profile,'TRITON_CACHE_DIR':environment['TRITON_CACHE_DIR']};assert len(full)==63 and profile['TIANSHU_FUTURE_KV_MODE']==('control' if batch==64 else 'candidate');checks.append(dict(scope=scope,batch=batch,source_count=len(sources),controller_count=len(controllers),environment_count=len(full),server_command=command,PYTHONPATH=profile['PYTHONPATH'],FutureKV=profile['TIANSHU_FUTURE_KV_MODE']))
gate=dict(complete=True,CPU_only=True,model_not_started=True,original877_sources_exact_subset=True,checks=checks,controller_sources=controllers,frozen_sources=sources0);(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules);print(json.dumps({k:v for k,v in gate.items() if k not in ('controller_sources','frozen_sources')}))
