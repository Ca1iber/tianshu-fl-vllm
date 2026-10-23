# SPDX-License-Identifier: Apache-2.0
import ast,importlib.util,json,sys
from pathlib import Path
from candidate_profile import BASE,EXP,prepare,digest,read
root=Path('/workspace/benchmark-results/20261023_dots_LM64_service_CPU_tianshu');assert not root.exists() and not (BASE/'CPU_GATE_32g.json').exists()
for p in BASE.rglob('*.py'):ast.parse(p.read_text())
root.mkdir();command,env,profile,sources=prepare(root,'candidate',check_gate=False);assert len(command)==13 and len(profile)==62 and len(profile|{'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']})==63
parent_root=Path('/workspace/benchmark-results/20261023_dots_Down64_v2_math8_tianshu');old=read(parent_root/'result.json');assert command==old['server_command'] and all(sources[p]==h for p,h in old['frozen_sources'].items()) and len(old['frozen_sources'])==916
from importlib.util import spec_from_file_location,module_from_spec
p=EXP/'tianshu_original_command_20261016/validation_profile.py';s=spec_from_file_location('_LM64_CPU_normalizer',p);m=module_from_spec(s);s.loader.exec_module(m)
a=m.normalized_environment(profile,root);b=m.normalized_environment(old['profile_environment'],parent_root);assert {k for k in a if a[k]!=b[k]}=={'PYTHONPATH'}
selection=read(BASE/'SELECTED_LM_32g.json');assert selection['payload_bytes']==142049280
for v in selection['buffers'].values():assert digest(v['file'])==v['file_sha256']
for name in ('inherited_C64_dense_audit.py','inherited_full_dense_audit.py'):
 s=spec_from_file_location('_LM64_CPU_'+name[:-3],BASE/name);module=module_from_spec(s);s.loader.exec_module(module)
assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,GPU_not_run=True,model_not_started=True,sources=sources,expected_model_source_count=len(sources)+1,parent916_model_sources_exact_subset=True,source_count_before_gate=len(sources),original13argv=True,environment_count=63,only_profile_difference='PYTHONPATH',LM64_only_new_precision='group128_symmetric_RTN_W4A16_dotsaddz',original_M8_and_otherM_and_GU_backend_delegated=True,selection_payload_bytes=142049280,selected_existing_component_bank_files_SHA_verified=True,explicit_full_and_C64_LM_guard_replacement_imports_CPU_only=True,complete_quality_controllers_not_yet_prepared=True,not_model_accuracy_or_speed_qualification=True)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
