# SPDX-License-Identifier: Apache-2.0
import ast,json,sys
from pathlib import Path
from candidate_profile import BASE,prepare,digest,frozen_check,normalized_environment
root=Path('/workspace/benchmark-results/20261022_native_int8_attention_short4k_CPU_retry1_tianshu');assert not root.exists();root.mkdir();arms={}
for path in BASE.glob('*.py'):ast.parse(path.read_text())
for arm in ('control','candidate'):
 out=root/arm;out.mkdir();command,env,profile,sources=prepare(out,arm,'prefill',check_quality=False)
 assert len(command)==13 and len(profile)==62 and len({**profile,'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']})==63
 frozen_check(sources);arms[arm]=dict(source_count=len(sources),sources=sources,profile=profile,root=str(out))
a=normalized_environment(arms['control']['profile'],arms['control']['root']);b=normalized_environment(arms['candidate']['profile'],arms['candidate']['root'])
assert {k for k in a if a[k]!=b[k]}=={'PYTHONPATH'}
assert arms['control']['source_count']==890 and arms['candidate']['source_count']==903
assert all(arms['candidate']['sources'][p]==h for p,h in arms['control']['sources'].items())
gate=dict(complete=True,CPU_only=True,model_not_started=True,fresh_quality_checked_at_CPU_prepare=False,runtime_requires_fresh_C8_C64_quality=True,baseline_model_sources=890,candidate_model_sources=903,original13argv=True,environment_count=63,controller_sources={str(p):digest(p) for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run_pair.sh']})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
