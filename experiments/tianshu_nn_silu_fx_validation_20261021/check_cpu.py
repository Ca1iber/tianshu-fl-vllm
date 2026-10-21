# SPDX-License-Identifier: Apache-2.0
import ast,json,os
os.environ['VLLM_PLUGINS']=''
from pathlib import Path
import torch
from framework_profile import BASE,SERVICE,prepare_runtime,read,digest
assert not torch.cuda.is_initialized()
for p in BASE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
root=Path('/workspace/benchmark-results/20261021_nn_silu_fx_validation_CPU_tianshu');assert not root.exists();root.mkdir()
command,environment,profile,sources=prepare_runtime(root,SERVICE)
assert len(sources)==450 and len(profile)==62 and len(command)==13
old=(BASE.parent/'tianshu_prefill_int8_validation_20261021/run_validation.py').read_text()
assert (BASE/'run_validation.py').read_text()==old.replace('selective_Grouped_INT8_prefill_mixed_and_NN_decode','NN_M64_FX_two_dot_GateUp_SiLU_and_static_compile64')
text=(BASE/'audit_profile.py').read_text();assert 'for m in (8,):' in text and "fusion['all_42_M64_real_capture_routes_verified']" in text
assert not torch.cuda.is_initialized()
controllers={str(p):digest(p) for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run_validation.sh']}
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,GPU_not_executed=True,runtime_source_count=450,environment_count=62,screened_runtime_sources_exactly_equal=True,official_evaluation_and_benchmark_commands_unchanged=True,C64_supplement_required_before_official=True,sources=dict(sources,**controllers))
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
