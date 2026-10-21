# SPDX-License-Identifier: Apache-2.0
import ast,json,os
os.environ['VLLM_PLUGINS']=''
from pathlib import Path
import torch
from auxiliary_profile import BASE,VALIDATION,SERVICE,MATH8,prepare_runtime,read,digest,normalized_environment
assert not torch.cuda.is_initialized()
for p in BASE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
root=Path('/workspace/benchmark-results/20261021_nn_silu_fx_C64_quality_CPU_v2_tianshu');assert not root.exists();root.mkdir()
command,environment,profile,sources=prepare_runtime(root,SERVICE)
canonical=read(MATH8/'state.json');assert sources==canonical['frozen_sources'] and len(sources)==450 and len(profile)==62 and command==canonical['server_command']
a=normalized_environment(profile,root);b=normalized_environment(canonical['profile_environment'],MATH8)
assert {k for k in a if a[k]!=b[k]}=={'TIANSHU_FUTURE_KV_MODE'} and a['TIANSHU_FUTURE_KV_MODE']=='control'
source=(BASE/'run_quality.py').read_text();assert '"--eval-batch-size", "64"' in source and "max_tokens\\\":32768" not in source
assert '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}' in source
policy=BASE.parent/'tianshu_future_kv_reservation_20261020/policy.py';tree=ast.parse(policy.read_text())
reservation=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Reservation');allocate=next(n for n in reservation.body if isinstance(n,ast.FunctionDef) and n.name=='allocate')
assert any(isinstance(n,ast.If) and ast.unparse(n.test)=="self.mode == 'candidate'" and any(isinstance(v,ast.Return) and isinstance(v.value,ast.Constant) and v.value.value is None for v in n.body) for n in ast.walk(allocate))
assert 'answer = self.original(*args, **kwargs)' in ast.unparse(allocate)
assert not torch.cuda.is_initialized()
controllers={str(p):digest(p) for p in [*BASE.glob('*.py'),*VALIDATION.glob('*.py'),*(BASE.parent/'tianshu_prefill_int8_validation_20261021').glob('*.py'),BASE.parent/'tianshu_prefill_int8_validation_20261021/batch_policy.json',BASE/'PLAN_32g.md',BASE/'run_after_math8.sh']}
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,GPU_not_executed=True,runtime_source_count=450,source_map_identical_to_canonical_Math8=True,only_FutureKV_environment_difference=True,all105_and_32768_output_preserved=True,native_allocator_call_retained_and_control_rejection_disabled_AST_verified=True,sources=dict(sources,**controllers))
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
