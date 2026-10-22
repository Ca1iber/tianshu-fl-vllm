# SPDX-License-Identifier: Apache-2.0
import ast,json,sys
from pathlib import Path
from quality_profile import BASE,HOLDOUT,prepare,read,digest,frozen_check,normalized_environment
for path in BASE.glob('*.py'):ast.parse(path.read_text())
root=Path('/workspace/benchmark-results/20261022_native_int8_attention_official4k_CPU_tianshu');assert not root.exists();root.mkdir()
command,environment,profile,sources=prepare(root,HOLDOUT,batch=8,scope='prefill',check_screen=False)
quality=read('/workspace/benchmark-results/20261022_native_int8_attention_math8_tianshu/accuracy_audit.json');assert sources==quality['frozen_sources'] and command==quality['server_command']
assert normalized_environment(profile,root)==normalized_environment(quality['profile_environment'],'/workspace/benchmark-results/20261022_native_int8_attention_math8_tianshu')
assert len(sources)==903 and len(command)==13 and len(profile)==62 and len({**profile,'TRITON_CACHE_DIR':environment['TRITON_CACHE_DIR']})==63
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
gate=dict(complete=True,CPU_only=True,model_not_started=True,official_controller_screen_not_yet_available=True,runtime_requires_positive_short_screen=True,fresh_C8_C64_quality_verified=True,source_count=903,environment_count=63,original13argv=True,controller_sources=read(root/'controller_sources.json'));(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='controller_sources'}))
