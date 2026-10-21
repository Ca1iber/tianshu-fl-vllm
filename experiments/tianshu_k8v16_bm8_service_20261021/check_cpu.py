# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,importlib.util,json,os,sys
from pathlib import Path
os.environ['VLLM_PLUGINS']=''
import torch
BASE=Path(__file__).resolve().parent
assert not torch.cuda.is_initialized()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p in BASE.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
selection=read(BASE/'selected_plan.json');gate=read(selection['component_gate']);assert gate['complete'] and gate['all_candidate_output_tensors_bitwise_equal_to_current'] and gate['full_output_elements_rechecked']==8044544
assert sha(selection['component_gate'])==selection['component_gate_sha256'] and gate['raw_result_sha256']==selection['component_raw_result_sha256'] and selection['selected_sequences']==[64] and selection['no_new_packed_weights']
for p,r in read(BASE/'generated_sources.json').items():assert sha(p)==sha(r['archive'])==r['sha256']
assert sha(BASE/'run_service_trial.py')==sha(BASE.parent/'tianshu_m_range_prefill_service_v2_20261021/run_service_trial.py')
s=importlib.util.spec_from_file_location('_BM8_CPU_prepare',BASE/'candidate_profile.py');profile=importlib.util.module_from_spec(s);s.loader.exec_module(profile)
root=Path('/workspace/benchmark-results/20261021_BM8_service_CPU_prepare_tianshu');root.mkdir(exist_ok=False);command,env,selected,sources=profile.prepare(root,'candidate')
old=read(BASE.parent/'tianshu_m_range_prefill_official_results_20261021/OFFICIAL_GATE_32g.json')['model_frozen_sources'];assert len(old)==471 and len(sources)>471 and all(sources[p]==h for p,h in old.items())
reference=read('/workspace/benchmark-results/20261021_qkv_NN_direct_reference_tianshu/result.json');assert reference['phase']=='complete' and reference['frozen_sources']==old
assert command==reference['server_command'] and len(command)==13 and len(selected)==62
normalized={k:v.replace(str(root),'__ROOT__') for k,v in selected.items()};other={k:v.replace(reference['output_root'],'__ROOT__') for k,v in reference['profile_environment'].items()};assert {k for k in normalized if normalized[k]!=other[k]}=={'PYTHONPATH'}
assert all(sha(p)==h==sha(root/'source_snapshot'/Path(p).relative_to('/')) for p,h in sources.items());assert not torch.cuda.is_initialized() and 'flag_gems' not in sys.modules
result=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,actual_entry_prepare_passed=True,original471_exact_subset=True,runtime_source_count=len(sources),environment_keys=62,original_argv_length=13,only_normalized_environment_difference='PYTHONPATH',different_source_graphs_explicitly_declared=True,client_byte_identical=True,no_new_packed_weights=True,component_all_outputs_bitwise_equal=True,component_full_rechecked_elements=8044544,component_timed_binaries=len(selection['timed_component_kernels']),four_generated_dependency_sources_archived=True,actual_model_Graph_capture_and_replay_required=True,new_Math_still_required=True,sources=sources)
assert not (BASE/'CPU_GATE_32g.json').exists();(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
