# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual new profile/source prepare without initializing CUDA."""
import ast,hashlib,importlib.util,json,os
from pathlib import Path
os.environ['VLLM_PLUGINS']=''
import torch
BASE=Path(__file__).resolve().parent
assert not torch.cuda.is_initialized()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
for p in BASE.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
selection=read(BASE/'selected_plan.json');component=read(selection['component_gate'])
assert sha(selection['component_gate'])==selection['component_gate_sha256'] and component['raw_result_sha256']==selection['component_raw_result_sha256']
assert component['complete'] and component['service_candidate_worth_testing'] and component['full_original_checkpoint_quantization_verified']
assert component['all_saved_INT32_elements_independently_CPU_recomputed']==9400320
assert selection['M']==64 and selection['configuration']==dict(bm=32,bn=128,bk=256,warps=2,stages=1)
assert selection['additional_weight_payload_bytes']==42*(2048*2560+2560*4)==220631040
assert sha(selection['kernel_source'])==selection['kernel_sha256']
old=BASE.parent/'tianshu_m_range_prefill_service_v2_20261021'
assert sha(BASE/'run_service_trial.py')==sha(old/'run_service_trial.py')
root=Path('/workspace/benchmark-results/20261021_QKV_NN_service_CPU_prepare_tianshu');root.mkdir(exist_ok=False)
profile=load('_QKV_NN_CPU_profile',BASE/'candidate_profile.py');command,env,selected,sources=profile.prepare(root,'candidate')
original=read(BASE.parent/'tianshu_m_range_prefill_official_results_20261021/OFFICIAL_GATE_32g.json')['model_frozen_sources']
assert len(original)==471 and len(sources)>471 and all(sources[p]==h for p,h in original.items())
reference=read('/workspace/benchmark-results/20261021_qkv_NN_direct_reference_tianshu/result.json')
assert reference['phase']=='complete' and reference['frozen_sources']==original
assert command==reference['server_command'] and len(command)==13 and len(selected)==62
normalized={k:v.replace(str(root),'__ARM_OUTPUT_ROOT__') for k,v in selected.items()}
normalized_reference={k:v.replace(reference['output_root'],'__ARM_OUTPUT_ROOT__') for k,v in reference['profile_environment'].items()}
assert sorted(k for k in normalized if normalized[k]!=normalized_reference[k])==['PYTHONPATH']
assert all(sha(p)==h==sha(root/'source_snapshot'/Path(p).relative_to('/')) for p,h in sources.items())
assert not torch.cuda.is_initialized() and 'flag_gems' not in __import__('sys').modules
report=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,actual_entry_prepare_passed=True,reference_completed=True,reference_source_count=471,runtime_source_count=len(sources),original471_exact_subset=True,different_source_graphs_explicitly_declared=True,only_normalized_environment_difference='PYTHONPATH',original_argv_length=13,environment_keys=62,client_source_byte_identical=True,component_actual_INT32_elements_independently_CPU_verified=9400320,extra_packed_weight_bytes=220631040,actual_capture_and_replay_still_require_GPU=True,new_Math_and_actual_C64_required=True,raw_root=str(root),sources=sources)
assert not (BASE/'CPU_GATE_32g.json').exists();(BASE/'CPU_GATE_32g.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))
