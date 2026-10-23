# SPDX-License-Identifier: Apache-2.0
import ast,json,sys
from pathlib import Path
import candidate_profile as p
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261023_RMS_A8_GU_service_CPU_tianshu');assert not ROOT.exists() and not (BASE/'CPU_GATE_32g.json').exists();ROOT.mkdir();original=p.read(BASE.parent/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');old=None;checks=[]
for path in BASE.rglob('*.py'):ast.parse(path.read_text())
for mode in ('control','candidate'):
 root=ROOT/mode;root.mkdir();command,environment,profile,sources=p.prepare(root,mode,False);assert all(sources[k]==v for k,v in original['frozen_sources'].items()) and command==original['server_command'];full={**profile,'TRITON_CACHE_DIR':environment['TRITON_CACHE_DIR']};assert len(full)==63;actual={k:v.replace(str(root),'<OUTPUT>') for k,v in full.items() if k!='PYTHONPATH'};target={k:v.replace(original['old_output_root'],'<OUTPUT>') for k,v in original['environment'].items() if k!='PYTHONPATH'};assert actual==target;assert old is None or sources==old;old=sources;checks.append(dict(mode=mode,source_count=len(sources),environment_count=len(full),original13_argv=command==original['server_command']))
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,original903_sources_exact_subset=True,sources=old,checks=checks);(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
