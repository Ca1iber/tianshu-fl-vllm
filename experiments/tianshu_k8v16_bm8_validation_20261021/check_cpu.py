# SPDX-License-Identifier: Apache-2.0
"""Exercise both actual497 quality prepares; no model or GPU initialization."""
import ast,hashlib,json,os,sys
from pathlib import Path
os.environ['VLLM_PLUGINS']=''
import torch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE))
assert not torch.cuda.is_initialized()
for p in BASE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
import framework_profile as canonical
import auxiliary_profile as auxiliary
root=Path('/workspace/benchmark-results/20261021_BM8_quality_CPU_prepare_tianshu');root.mkdir(exist_ok=False)
answers={}
for arm,module in [('C8',canonical),('C64',auxiliary)]:
 r=root/arm;r.mkdir();command,env,profile,sources=module.prepare_runtime(r,canonical.SCREEN,require_screen=False)
 answers[arm]=dict(command=command,profile=canonical.normalized_environment(profile,r),sources=sources)
 assert sources==canonical.read(BASE.parent/'tianshu_k8v16_bm8_service_20261021/CPU_GATE_32g.json')['sources']
assert answers['C8']['command']==answers['C64']['command'] and answers['C8']['sources']==answers['C64']['sources']
assert {k for k in answers['C8']['profile'] if answers['C8']['profile'][k]!=answers['C64']['profile'][k]}=={'TIANSHU_FUTURE_KV_MODE'}
assert len(answers['C8']['sources'])==497 and len(answers['C8']['profile'])==62 and len(answers['C8']['command'])==13
frozen=dict(answers['C8']['sources']);frozen.update({str(p):canonical.digest(p) for p in [*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'PLAN_32g.md']});canonical.frozen_check(frozen)
assert not torch.cuda.is_initialized() and 'flag_gems' not in sys.modules
result=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,actual_C8_and_C64_prepare_exercised=True,runtime_source_count=497,screened497_source_map_exactly_retained=True,only_C64_admission_environment_difference=True,original_argv_length=13,environment_keys=62,performance_screen_required_before_GPU=True,fresh_Math8_and_actual_C64_required=True,old_Math_accuracy_not_reused=True,raw_root=str(root),sources=frozen)
assert not (BASE/'CPU_GATE_32g.json').exists();(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
