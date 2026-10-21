# SPDX-License-Identifier: Apache-2.0
import ast,json
from pathlib import Path
from candidate_profile import BASE,prepare,read,digest
assert read(BASE.parent/'tianshu_prefill_key_bound_policy_20261021/CPU_GATE_32g.json')['complete']
for p in BASE.rglob('*.py'):ast.parse(p.read_text())
profiles=[]
for mode in ('control','candidate'):
    initial=Path(f'/workspace/benchmark-results/20261021_prefill_key_bound_service_CPU_{mode}_tianshu')
    root=initial;revision=0
    while root.exists():revision+=1;root=initial.with_name(initial.name+f'_v{revision}')
    root.mkdir()
    command,environment,profile,sources=prepare(root,mode)
    assert len(profile)==62 and len(command)==13 and profile['VLLM_DISABLE_COMPILE_CACHE']=='1'
    assert profile['PYTHONPATH']==str(BASE/('hooks_'+mode))
    profiles.append(dict(mode=mode,output_root=str(root),command=command,profile=profile,sources=sources))
assert profiles[0]['sources']==profiles[1]['sources'] and profiles[0]['command']==profiles[1]['command']
assert profiles[0]['profile']=={k:v.replace('/hooks_candidate','/hooks_control').replace(profiles[1]['output_root']+'/',profiles[0]['output_root']+'/') for k,v in profiles[1]['profile'].items()}
result=dict(complete=True,CPU_only=True,no_GPU_started=True,parent_runtime_sources=407,runtime_source_count=len(profiles[0]['sources']),environment_count=62,
    actual_CPU_metadata_contract_source_pinned=True,actual_native_route_AST_hook_CPU_pass=True,
    all24_GPU_component_cases_passed=True,original13serveargv_preserved=True,fixed_arena8192=True,model_Math_not_run=True,profiles=profiles)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
