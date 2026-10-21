# SPDX-License-Identifier: Apache-2.0
"""Run actual profile preparation for both validation modes without CUDA."""
import ast,hashlib,json,os,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',VLLM_PLUGINS='',PYTHONPATH='')
import torch
import framework_profile as f
import auxiliary_profile as a
BASE=Path(__file__).resolve().parent
root=Path('/workspace/benchmark-results/20261021_m_range_prefill_validation_CPU_prepare_tianshu')
assert not root.exists() and not torch.cuda.is_initialized();root.mkdir()
prepared=[]
for name,module in (('Math8',f),('C64',a)):
 arm=root/name;arm.mkdir();command,env,profile,sources=module.prepare_runtime(arm,f.SCREEN,require_screen=False)
 assert len(sources)==471 and len(profile)==62 and len(command)==13
 f.frozen_check(sources)
 original=f.read(BASE.parent/'tianshu_nn_silu_fx_c64_quality_20261021/C64_QUALITY_GATE_32g.json')['model_frozen_sources'];assert len(original)==450 and all(sources[p]==h for p,h in original.items())
 prepared.append(dict(mode=name,command=command,environment=profile,sources=sources))
assert prepared[0]['sources']==prepared[1]['sources'] and prepared[0]['command']==prepared[1]['command']
envs=[{k:v.replace(str(root/r['mode']),'__ARM_OUTPUT_ROOT__') for k,v in r['environment'].items()} for r in prepared]
assert {k for k in envs[0] if envs[0][k]!=envs[1][k]}=={'TIANSHU_FUTURE_KV_MODE'}
assert envs[0]['TIANSHU_FUTURE_KV_MODE']=='candidate' and envs[1]['TIANSHU_FUTURE_KV_MODE']=='control'
for p in BASE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
sys.modules.pop('sitecustomize',None)
assert not torch.cuda.is_initialized() and not any(k.startswith(('flag_gems','vllm')) for k in sys.modules)
assert f.digest('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')==f.BENCH_SHA
for name,batch in (('run_validation.py','8'),('run_c64_quality.py','64')):
 tree=ast.parse((BASE/name).read_text());matches=[n for n in ast.walk(tree) if isinstance(n,ast.List) and any(isinstance(e,ast.Constant) and e.value=='--eval-batch-size' for e in n.elts)];assert len(matches)==1
 values=[e.value if isinstance(e,ast.Constant) else None for e in matches[0].elts];assert values[values.index('--eval-batch-size')+1]==batch
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'PLAN_32g.md']
checks={str(p):f.digest(p) for p in paths};checks.update(prepared[0]['sources'])
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,actual_prepare_modes=['Math8','C64'],runtime_source_count=471,environment_count=62,original_serve_arguments=13,original450_exact_subset=True,only_C64_environment_difference='TIANSHU_FUTURE_KV_MODE',GPU_perf_gate_still_required=True,actual_prepare_root=str(root),sources=checks)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(root/'prepared_modes.json').write_text(json.dumps(prepared,indent=2)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
