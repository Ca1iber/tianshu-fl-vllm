# SPDX-License-Identifier: Apache-2.0
"""Bind both independently qualified fitted components to the unchanged r4."""
import importlib.util
from pathlib import Path
import statistics
BASE=Path(__file__).resolve().parent
EXP=BASE.parent
spec=importlib.util.spec_from_file_location('_QKVO64_r4_parent',EXP/'tianshu_W4_LMpad_service_v2_20261023/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
COMPONENTS={r:EXP/f'tianshu_W4_{r}64_fitted_component_20261024' for r in ('O','QKV')}

def merge(sources,extra):
    for p,h in extra.items():
        assert p not in sources or sources[p]==h,(p,'conflicting source')
        sources[p]=h

def component_gate(role):
    folder=COMPONENTS[role]
    cpu=read(folder/'CPU_GATE_32g.json');gpu=read(folder/'INDEPENDENT_CPU_GATE_32g.json')
    assert cpu['complete'] and cpu['all42_original_weights_SHA_match'] and cpu['all84_full_pack_roundtrips_passed']
    assert gpu['complete'] and gpu['all_sources_files_SHA_verified'] and gpu['all_full_FP64_numeric_passed']
    assert gpu['all42_determinism_guard_and_byte_scope'] and gpu['qualification'][role]['W4_fitted']
    assert digest(Path(gpu['raw_directory'])/'result.json')==gpu['result_sha256']
    rows={r['variant']:r for r in gpu['timings'] if r['mode']=='graph' and r['role']==role}
    fitted,baseline,minmax=(rows[v] for v in ('W4_fitted','BF16_actual_parent','W4_minmax'))
    assert fitted['performance_qualified'] and fitted['component_advancement_gate']
    assert fitted['spread_percent']<1 and baseline['spread_percent']<1
    assert fitted['median_42banks_ms']<=.95*baseline['median_42banks_ms']
    assert fitted['median_42banks_ms']<=1.01*minmax['median_42banks_ms']
    errors={v:[r['relative_L2_vs_original_FP64'] for r in gpu['projection_errors'] if r['role']==role and r['variant']==v] for v in ('W4_fitted','W4_minmax')}
    assert len(errors['W4_fitted'])==len(errors['W4_minmax'])==9
    assert statistics.mean(errors['W4_fitted'])<statistics.mean(errors['W4_minmax'])
    return cpu,gpu

def prepare(root,mode='candidate',check_gate=True):
    assert mode in ('control','candidate')
    command,env,profile,sources=parent.prepare(root,'candidate')
    assert len(sources)==944
    old_sources=dict(sources);old_profile=dict(profile)
    merge(sources,read(BASE/'CORE_CPU_GATE_32g.json')['sources'])
    selection=read(BASE/'SELECTED_BANKS_32g.json')
    assert selection['complete'] and selection['group_size']==128 and selection['format']=='fitted_affine'
    selected={(b['role'],b['layer']):b for b in selection['banks']}
    assert len(selected)==84 and selection['additional_payload_bytes']==210567168
    for role,folder in COMPONENTS.items():
        cpu,gpu=component_gate(role)
        assert cpu['checkpoint_sha256']=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
        merge(sources,cpu['sources'])
        for b in cpu['banks']:
            actual=selected[(role,b['layer'])];variant=b['variants']['fitted']
            assert actual['original_sha256']==b['original_sha256'] and actual['N']==b['N'] and actual['K']==b['K']==2048
            for key in ('packed_sha256','scale_sha256','zero_sha256','payload_bytes'):assert actual[key]==variant[key]
        for name in ('CPU_GATE_32g.json','INDEPENDENT_CPU_GATE_32g.json','EXECUTION.sha256'):
            p=folder/name;sources[str(p)]=digest(p)
    paths=set(BASE.rglob('*.py'))|{BASE/'PLAN_32g.md',BASE/'SELECTED_BANKS_32g.json',BASE/'CORE_CPU_GATE_32g.json',BASE/'CHECKPOINT_CPU.sha256',EXP/'tianshu_W4_O64_affine_service_20261024/audit.py'}
    paths |= set(BASE.glob("*.sh"))
    merge(sources,{str(p):digest(p) for p in paths})
    frozen_check(sources)
    assert all(sources[p]==h for p,h in old_sources.items())
    if check_gate:
        gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'] and gate['model_ready'] and gate['sources']==sources
        sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
    profile.update(PYTHONPATH=str(BASE/'hooks_candidate'),TIANSHU_QKVO64_FITTED_MODE=mode,
        TIANSHU_QKVO64_FITTED_RECORD_DIR=str(root/'QKVO64_fitted_records'))
    assert all(profile[k]==v for k,v in old_profile.items() if k!='PYTHONPATH')
    env.update(profile)
    for p,h in sources.items():
        target=root/'source_snapshot'/Path(p).relative_to('/')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes())
        assert digest(target)==h
    return command,env,profile,sources
