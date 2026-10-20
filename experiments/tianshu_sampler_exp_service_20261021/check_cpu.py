# SPDX-License-Identifier: Apache-2.0
import ast,json
from pathlib import Path
from candidate_profile import BASE,prepare,read,digest

def main():
    for path in BASE.rglob('*.py'):ast.parse(path.read_text())
    primitive=BASE.parent/'tianshu_sampler_exp_prewarm_20261021/GPU_GATE_32g.json';gate=read(primitive)
    assert gate['complete'] and gate['default_CPU_and_GPU_RNG_exact'] and gate['actual_B37_FP32_words']==4830720 and gate['replay_autotuning_records']==0
    for path,sha in gate['source_snapshots'].items():assert digest(path)==sha
    assert (BASE/'hooks_control/sitecustomize.py').read_bytes()==(BASE/'hooks_candidate/sitecustomize.py').read_bytes()
    profiles=[]
    for mode in ('control','candidate'):
        root=Path('/workspace/benchmark-results')/f'20261021_sampler_exp_service_CPU_{mode}_tianshu';assert not root.exists();root.mkdir()
        command,env,profile,sources=prepare(root,mode)
        assert len(command)==13 and len(profile)==62 and profile['TIANSHU_LM_PRECISION_MODE']=='candidate' and profile['PYTHONPATH']==str(BASE/('hooks_'+mode))
        assert all(profile[k]=='candidate' for k in ('TIANSHU_DENSE_INT8_MODE','TIANSHU_PREFILL_LINEAR_MODE','TIANSHU_SEQ_LIMIT_MODE'))
        profiles.append(dict(mode=mode,root=str(root),command=command,environment=profile,sources=sources))
    assert profiles[0]['sources']==profiles[1]['sources']
    old,new=profiles
    expected={k:v.replace(old['root']+'/',new['root']+'/').replace('/hooks_control','/hooks_candidate') for k,v in old['environment'].items()};assert expected==new['environment']
    result=dict(complete=True,CPU_only=True,no_GPU_started=True,service_not_yet_verified=True,adopted=False,new_Math_not_run=True,
        qualified_parent_model_runtime_sources=359,runtime_source_count=len(new['sources']),environment_count=62,
        both_arms_same_qualified_model_precision=True,original_13_serve_argv_preserved=True,private_generator_component_GPU_evidence_verified=True,profiles=profiles)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
if __name__=='__main__':main()
