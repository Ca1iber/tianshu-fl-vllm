# SPDX-License-Identifier: Apache-2.0
"""Keep the full released r5 and add only the separately qualified BM8 child."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
EXP=BASE.parent
R5=EXP/'tianshu_W4_QKVO64_fitted_service_20261024'
BM8=EXP/'tianshu_k8v16_bm8_service_v3_20261021'
spec=importlib.util.spec_from_file_location('_r5_BM8_unchanged_parent',R5/'candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check,merge=parent.digest,parent.read,parent.frozen_check,parent.merge
PAIR_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'

def released_sources():
    result={}
    for line in (EXP/'tianshu_W4_QKVO64_fitted_release_20261024/MODEL_SOURCES.sha256').read_text().splitlines():
        h,p=line.split(None,1);p=p.lstrip('* ')
        assert p.startswith('/') and len(h)==64 and p not in result
        result[p]=h
    assert len(result)==997
    return result

def prepare(root,mode='candidate',check_gate=True):
    assert mode in ('candidate','control')
    # Parent control would disable fitted QKV/O W4 and produce the wrong baseline.
    command,env,profile,sources=parent.prepare(root,'candidate')
    released=released_sources()
    assert sources==released,'The comparison must start from the exact released r5'
    assert profile['TIANSHU_QKVO64_FITTED_MODE']=='candidate' and len(profile)+1==65
    selection=read(BM8/'selected_plan.json')
    assert selection['complete'] and selection['selected_sequences']==[64]
    gate=read(selection['component_gate'])
    assert digest(selection['component_gate'])==selection['component_gate_sha256']
    assert gate['complete'] and gate['all_candidate_output_tensors_bitwise_equal_to_current']
    assert gate['full_output_elements_rechecked']==8044544
    assert digest(Path(gate['raw_root'])/'result.json')==gate['raw_result_sha256']==selection['component_raw_result_sha256']
    assert selection['kernel_sha256']=='5cc0553c18a6aa3696b6bfb7e2f9433c45469cf4b4a1bb5eef806c5c1b5510a9'
    assert selection['BLOCK_M']==8 and selection['BLOCK_Q']==1 and selection['TILE_SIZE']==16
    assert selection['num_warps']==selection['num_stages']==1
    legacy=read(EXP/'tianshu_m_range_prefill_official_results_20261021/OFFICIAL_GATE_32g.json')['model_frozen_sources']
    assert len(legacy)==471 and all(sources[p]==h for p,h in legacy.items())
    merge(sources,gate['frozen_sources'])
    merge(sources,{v['generated_source']:v['generated_sha256'] for v in gate['model_sources'].values()})
    for path_key,hash_key in [('component_gate','component_gate_sha256'),('kernel_source','kernel_sha256'),
        ('binary_signature_source','binary_signature_sha256'),('binary_code_gate','binary_code_gate_sha256'),
        ('recovery_verifier_source','recovery_verifier_sha256')]:
        merge(sources,{selection[path_key]:selection[hash_key]})
    paths=set(BASE.rglob('*.py'))|{BASE/'PLAN_32g.md',BM8/'policy.py',BM8/'audit_profile.py',BM8/'selected_plan.json'}
    merge(sources,{str(p):digest(p) for p in paths})
    frozen_check(sources)
    if check_gate:
        cpu=read(BASE/'CPU_GATE_32g.json')
        assert cpu['complete'] and cpu['model_ready'] and cpu['sources']==sources
        sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
    for p,h in sources.items():
        target=root/'source_snapshot'/Path(p).relative_to('/')
        if p in released:
            assert target.exists() and digest(target)==h
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.is_symlink()
            target.write_bytes(Path(p).read_bytes())
            assert digest(target)==h
    profile.update(PYTHONPATH=str(BASE/'hooks_candidate'),TIANSHU_R5_BM8_MODE=mode,
                   TIANSHU_R5_BM8_RECORD_DIR=str(root/'bm8_decode_records'))
    assert profile['TIANSHU_QKVO64_FITTED_MODE']=='candidate'
    env.update(profile)
    return command,env,profile,sources
