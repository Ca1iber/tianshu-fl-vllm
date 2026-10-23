# SPDX-License-Identifier: Apache-2.0
import ast,importlib.util,sys
from pathlib import Path
from candidate_profile import BASE,EXP,prepare,digest,read
root=Path('/workspace/benchmark-results/20261023_W4_LMpad_service_v2_CPU_tianshu');assert not root.exists() and not (BASE/'CPU_GATE_32g.json').exists();root.mkdir()
for p in BASE.rglob('*.py'):ast.parse(p.read_text())
command,env,profile,sources=prepare(root,'candidate',check_gate=False);assert len(command)==13 and len(profile)==62 and len(profile|{'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']})==63
oldroot=Path('/workspace/benchmark-results/20261023_dots_Down64_v2_math8_tianshu');old=read(oldroot/'result.json');assert command==old['server_command'] and all(sources[p]==h for p,h in old['frozen_sources'].items()) and len(old['frozen_sources'])==916
p=EXP/'tianshu_original_command_20261016/validation_profile.py';s=importlib.util.spec_from_file_location('_LMpad_normalizer',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);a=m.normalized_environment(profile,root);b=m.normalized_environment(old['profile_environment'],oldroot);assert {k for k in a if a[k]!=b[k]}=={'PYTHONPATH'}
selection=read(BASE/'SELECTED_LM_32g.json');assert selection['payload_bytes']==142049280
for v in selection['buffers'].values():assert digest(v['file'])==v['file_sha256']
for name in ('inherited_C8_dense_audit.py','inherited_C64_dense_audit.py','inherited_full_dense_audit.py','inherited_down_audit.py'):
 s=importlib.util.spec_from_file_location('_LMpad_check_'+name[:-3],BASE/name);module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
pad=EXP/'tianshu_W4_LM_pad32_64_component_v2_20261023'
for p,h in read(pad/'CPU_GATE_32g.json')['sources'].items():assert digest(p)==h
for p,h in read(pad/'CPU_GATE_32g.json')['files'].items():assert digest(p)==h
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,GPU_not_run=True,sources=sources,expected_model_source_count=len(sources)+1,original13argv=True,environment_count=63,only_profile_difference='PYTHONPATH',all_model_sources_916_parent_subset=True,new_LM1_64_W4_direct1_2_and_pad32_64=True,fresh_complete_quality_required=True,explicit_C8_C64_full_and_Down_audit_adapters_imported=True,original_checkpoint_not_changed=True)
(BASE/'CPU_GATE_32g.json').write_text(__import__('json').dumps(gate,indent=2)+'\n');print(__import__('json').dumps({k:v for k,v in gate.items() if k!='sources'}))
