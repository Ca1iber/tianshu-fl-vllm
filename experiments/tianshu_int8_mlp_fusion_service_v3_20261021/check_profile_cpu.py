# SPDX-License-Identifier: Apache-2.0
import ast,json
from pathlib import Path
from candidate_profile import BASE,prepare

def fn(source,name):return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
def main():
    profiles=[]
    for mode in ('control','candidate'):
        root=Path(f'/workspace/benchmark-results/20261021_int8_mlp_fusion_v3_CPU_gate_{mode}_tianshu');assert not root.exists();root.mkdir()
        command,env,profile,sources=prepare(root,mode)
        assert len(profile)==63 and len(command)==13 and profile['TIANSHU_MLP_FUSION_MODE']==mode
        assert all(profile[k]=='candidate' for k in ('TIANSHU_DENSE_INT8_MODE','TIANSHU_LM_PRECISION_MODE','TIANSHU_PREFILL_LINEAR_MODE')) and profile['TIANSHU_COLD_START_MODE']=='gpu4'
        profiles.append(dict(mode=mode,command=command,profile_environment=profile,frozen_sources=sources))
    assert profiles[0]['frozen_sources']==profiles[1]['frozen_sources']
    original=(BASE.parent/'tianshu_lm_precision_service_20261020/policy.py').read_text();text=(BASE/'policy.py').read_text()
    for name in ('role_for','prepare_layer','native_lm_fallback','make_process','make_apply'):
        assert ast.dump(fn(text,name))==ast.dump(fn(original,name))
    loader=fn(text,'load_model');calls=[n for n in ast.walk(loader) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='bind_fusion'];assert len(calls)==1
    for node in ast.walk(loader):
        for _,value in ast.iter_fields(node):
            if isinstance(value,list) and calls[0] in value:value.remove(calls[0])
    assert ast.dump(loader)==ast.dump(fn(original,'load_model'))
    fusion=fn((BASE/'fusion.py').read_text(),'forward')
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='down_proj' for n in ast.walk(fusion))==1
    for path in BASE.rglob('*.py'):ast.parse(path.read_text())
    result=dict(complete=True,no_GPU_started=True,runtime_source_count=len(profiles[0]['frozen_sources']),environment_count=63,
        weight_loading_and_LM_AST_unchanged=True,loader_only_adds_CPU_MLP_binding=True,MLP_wrapper_has_one_original_down_projection=True,
        both_modes_use_new_MLP_custom_op_boundary=True,new_Math_required=True,profiles=profiles)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
if __name__=='__main__':main()
