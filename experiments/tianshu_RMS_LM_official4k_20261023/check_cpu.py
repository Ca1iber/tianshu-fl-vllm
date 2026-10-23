# SPDX-License-Identifier: Apache-2.0
import ast,json,sys
from pathlib import Path
from quality_profile import BASE,HOLDOUT,prepare,read,digest
for p in BASE.glob('*.py'):ast.parse(p.read_text())
root=Path('/workspace/benchmark-results/20261023_RMS_LM_official4k_CPU_tianshu');assert not root.exists();root.mkdir();cmd,env,profile,sources=prepare(root,HOLDOUT,batch=8,scope='prefill',check_screen=False,check_quality=False);assert len(sources)==944 and len(cmd)==13 and len(profile)==62 and len({**profile,'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']})==63;assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
gate=dict(complete=True,CPU_only=True,model_not_started=True,source_count=944,original13argv=True,environment_count=63,new_C64_C8_quality_and_short_pending=True,runtime_requires_both_new_quality_and_positive_short=True,controller_sources=read(root/'controller_sources.json'));(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='controller_sources'}))
