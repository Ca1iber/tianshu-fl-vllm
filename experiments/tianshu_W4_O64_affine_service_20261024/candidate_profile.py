# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
EXP=BASE.parent
spec=importlib.util.spec_from_file_location('_O64_r4_parent',EXP/'tianshu_W4_LMpad_service_v2_20261023/candidate_profile.py')
parent=importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode='candidate',check_gate=True):
    assert mode in ('control','candidate')
    command,env,profile,sources=parent.prepare(root,'candidate')
    assert len(sources)==944
    old_sources=dict(sources)
    old_profile=dict(profile)
    component=EXP/'tianshu_W4_attention64_20261024'
    proof=read(component/'INDEPENDENT_CPU_GATE_32g.json')
    assert proof['complete'] and proof['all_full_FP64_numeric_passed']
    assert digest(Path(proof['raw_directory'])/'result.json')==proof['result_sha256']
    paths=set(BASE.rglob('*.py'))|{BASE/'PLAN_32g.md',BASE/'SELECTED_O_32g.json'}
    paths|={component/n for n in ('common.py','PLAN_32g.md','CPU_GATE_32g.json','INDEPENDENT_CPU_GATE_32g.json','EXECUTION.sha256')}
    sources.update({str(p):digest(p) for p in paths})
    frozen_check(sources)
    assert all(sources[p]==h for p,h in old_sources.items())
    if check_gate:
        gate=read(BASE/'CPU_GATE_32g.json')
        assert gate['complete'] and gate['sources']==sources
        sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
    profile.update(PYTHONPATH=str(BASE/'hooks_candidate'),TIANSHU_O64_AFFINE_MODE=mode,
        TIANSHU_O64_AFFINE_RECORD_DIR=str(root/'O64_affine_records'))
    assert all(profile[k]==v for k,v in old_profile.items() if k!='PYTHONPATH')
    env.update(profile)
    for p,h in sources.items():
        target=root/'source_snapshot'/Path(p).relative_to('/')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(p).read_bytes())
        assert digest(target)==h
    return command,env,profile,sources
