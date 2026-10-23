# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
parent=load('_prefill_RMS_QKV_exact_r3',EXP/'tianshu_native_int8_attention_service_20261022/candidate_profile.py');digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode='candidate',check_gate=True):
 assert mode in ('control','candidate');command,environment,profile,sources=parent.prepare(root,'candidate');published=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');assert len(sources)==903 and sources==published['frozen_sources'] and command==published['server_command'];original=dict(sources);old_profile=dict(profile);component=EXP/'tianshu_prefill_RMS_A8_QKV_20261023';gate=read(component/'INDEPENDENT_CPU_GATE_32g.json');preparation=read(component/'CPU_PREPARE_GATE_32g.json');assert gate['complete'] and gate['component_lead'] and gate['numeric_cases']==18 and gate['full_K_FP64_norm_cells']==84934656 and gate['full_K_INT64_projection_reference_cells']==4608 and gate['all_sources_and_snapshots_SHA_verified'];assert gate['verifier_sha256']==digest(component/'verify_cpu.py') and preparation['fixed_norm_kernel_sha256']==digest(EXP/'tianshu_IR_RMS_A8_20261023/kernel.py')
 paths=set(BASE.rglob('*.py'))|{BASE/'PLAN_32g.md',BASE/'PARENT_BINDING_CPU_32g.json',component/'INDEPENDENT_CPU_GATE_32g.json',component/'CPU_PREPARE_GATE_32g.json',component/'GPU_INPUTS.sha256',EXP/'tianshu_IR_RMS_A8_20261023/kernel.py',Path('/usr/local/lib/python3.12/site-packages/torch/_library/custom_ops.py'),Path('/workspace/MiniCPM5-2B/config.json')}
 for line in (component/'GPU_INPUTS.sha256').read_text().splitlines():
  expected,path=line.split('  ',1)
  if Path(path).suffix=='.py':assert digest(path)==expected;paths.add(Path(path))
 sources.update({str(path):digest(path) for path in paths});frozen_check(sources);assert all(sources[path]==h for path,h in original.items())
 if check_gate:own=read(BASE/'CPU_GATE_32g.json');assert own['complete'] and own['sources']==sources;sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
 profile['PYTHONPATH']=str(BASE/f'hooks_{mode}');environment.update(profile);assert all(profile[key]==value for key,value in old_profile.items() if key!='PYTHONPATH')
 for path,expected in sources.items():
  target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert digest(target)==expected
 return command,environment,profile,sources
