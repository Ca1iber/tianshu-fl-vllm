# SPDX-License-Identifier: Apache-2.0
import ast,sys
from pathlib import Path
import candidate_profile as p
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261023_dots_Down64_service_v2_CPU_tianshu');assert not ROOT.exists() and not (BASE/'CPU_GATE_32g.json').exists();ROOT.mkdir();release=p.read(BASE.parent/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json')
for path in BASE.rglob('*.py'):ast.parse(path.read_text())
command,env,profile,sources=p.prepare(ROOT,check_gate=False);assert command==release['server_command'] and all(sources[x]==h for x,h in release['frozen_sources'].items());full={**profile,'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']};assert len(full)==63
actual={k:v.replace(str(ROOT),'<OUTPUT>') for k,v in full.items() if k!='PYTHONPATH'};expected={k:v.replace(release['old_output_root'],'<OUTPUT>') for k,v in release['environment'].items() if k!='PYTHONPATH'};assert actual==expected
# Preserve every old Down audit condition except the explicit M64 BF16 capture requirement.
old=(BASE.parent/'tianshu_native_int8_down_service_v2_20261022/audit_profile.py').read_text();new=(BASE/'inherited_down_audit.py').read_text();transformed=old.replace('BASE=Path(__file__).resolve().parent',"BASE=Path(__file__).resolve().parent.parent/'tianshu_native_int8_down_service_v2_20261022'").replace("assert len({r['bank_id'] for r in capture})==42",'assert not capture').replace("assert all(r['implementation']==('native_A8W8_Down' if record['scope']=='all' else 'original_BF16_Down') for r in capture)",'assert not capture');assert new==transformed
assert not any(k.split('.')[0] in ('torch','triton','vllm','flag_gems') for k in sys.modules)
g=dict(complete=True,CPU_only=True,no_GPU_imports=True,original903_sources_exact_subset=True,server_command=command,environment_count=63,only_environment_change='PYTHONPATH',old_Down_audit_all_other_conditions_byte_exact=True,sources=sources,expected_model_source_count=len(sources)+1);(BASE/'CPU_GATE_32g.json').write_text(__import__('json').dumps(g,indent=2)+'\n');print(__import__('json').dumps({k:v for k,v in g.items() if k!='sources'}))
