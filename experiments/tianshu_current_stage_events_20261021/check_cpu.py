# SPDX-License-Identifier: Apache-2.0
import ast,json,os,sys
from pathlib import Path
os.environ['VLLM_PLUGINS']=''
import torch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE));assert not torch.cuda.is_initialized()
for p in BASE.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
from candidate_profile import prepare,digest,frozen_check
root=Path('/workspace/benchmark-results/20261021_current_stage_events_CPU_prepare_tianshu');root.mkdir(exist_ok=False)
command,env,profile,sources=prepare(root,None)
profile.update(PYTHONPATH=str(BASE/'hooks_candidate'),TIANSHU_STAGE_PROFILE_PROBE='1',TIANSHU_STAGE_PROFILE_MODE='events',TIANSHU_STAGE_PROFILE_RECORD_DIR=str(root/'stage_records'),TIANSHU_STAGE_PROFILE_ARM_FILE=str(root/'profile_armed'))
assert len(profile)==66 and len(command)==13
policy=(BASE/'policy.py').read_text();assert "for _ in range(40)" in policy and policy.count('torch.cuda.synchronize()')==2 and "state['warm']<=32" in policy
frozen_check(sources);assert not torch.cuda.is_initialized()
result=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,actual_prepare_passed=True,original471_exact_subset=True,diagnostic_source_count=len(sources),environment_keys=66,original_argv_length=13,event_window_frames=40,Graph_external_events=True,two_boundary_synchronizations=True,not_benchmark_or_quality=True,sources=sources)
assert not (BASE/'CPU_GATE_32g.json').exists();(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
