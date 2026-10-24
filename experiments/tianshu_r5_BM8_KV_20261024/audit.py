# SPDX-License-Identifier: Apache-2.0
"""Full r5 accounting plus actual final BM8 JIT, binary, capture and replay."""
import importlib.util,sys
from pathlib import Path
from candidate_profile import BASE,EXP,BM8,PAIR_SHA,digest,frozen_check,read,released_sources

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
parent=load('_r5_BM8_full_r5_audit',EXP/'tianshu_W4_QKVO64_fitted_service_20261024/audit.py')
extension=load('_r5_BM8_actual_final_code_audit',BM8/'audit_profile.py')
audit_math=parent.audit_math

def audit_model(root,state,runtime=False):
    assert state['mode']=='candidate' and state['bm8_mode'] in ('candidate','control')
    assert state['profile_environment']['TIANSHU_QKVO64_FITTED_MODE']=='candidate'
    assert state['profile_environment']['TIANSHU_R5_BM8_MODE']==state['bm8_mode']
    released=released_sources()
    assert all(state['frozen_sources'][p]==h for p,h in released.items())
    result=parent.audit_model(root,state,runtime=runtime)
    proof=read(root/'bm8_integration.json')
    assert proof['mode']==state['bm8_mode'] and proof['parent_QKVO_mode']=='candidate' and proof['parent_full_r5']
    assert proof['source_sha256']==state['frozen_sources'][proof['source']]
    pair=list((root/'pair_load_records').glob('binding_*.json'));assert len(pair)==1
    binding=read(pair[0])
    assert binding['mode']=='candidate' and binding['installed'] and binding['candidate_sha256']==PAIR_SHA
    if state['bm8_mode']=='candidate':
        assert proof['BM8_installed']
        result=extension.audit_extension(root,state,result,require_C64=(runtime and state['batch']==64))
    else:
        assert not proof['BM8_installed'] and not (root/'bm8_decode_records').exists()
        rows=[r for r in binding['launches'].values() if r['inside_capture'] and r['sequences']==64 and r['IS_3D'] and r['selected_pair_rows']]
        assert sum(r['calls'] for r in rows)>=42
        assert all(r['active_source_sha256']==PAIR_SHA and r['grid']==[96,2,4] and r['n_spills']==0 for r in rows)
        result.update(active_decode_source_sha256=PAIR_SHA,control_r5_pair_capture_verified=True)
    result.update(original997_exact_subset=True,bm8_mode=state['bm8_mode'],r5_QKVO_fitted_preserved=True,
                  BM8_integration_proof=proof,runtime_source_count=len(state['frozen_sources']))
    return result
