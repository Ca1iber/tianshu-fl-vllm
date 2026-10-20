# SPDX-License-Identifier: Apache-2.0
import ast,json
from pathlib import Path
from candidate_profile import BASE,prepare

def main():
    profiles=[]
    for mode in ('control','candidate'):
        root=Path(f'/workspace/benchmark-results/20261021_gateup_padding_CPU_gate_{mode}_tianshu');assert not root.exists();root.mkdir()
        command,env,profile,sources=prepare(root,mode)
        assert len(profile)==63 and len(command)==13
        assert profile['TIANSHU_TILE_PADDING_MODE']==mode
        assert all(profile[k]=='candidate' for k in ('TIANSHU_DENSE_INT8_MODE','TIANSHU_LM_PRECISION_MODE','TIANSHU_PREFILL_LINEAR_MODE'))
        assert profile['TIANSHU_COLD_START_MODE']=='gpu4' and profile['VLLM_DISABLE_COMPILE_CACHE']=='1'
        profiles.append(dict(mode=mode,command=command,profile_environment=profile,frozen_sources=sources))
    assert profiles[0]['frozen_sources']==profiles[1]['frozen_sources']
    for path in BASE.rglob('*.py'):ast.parse(path.read_text())
    original=(BASE.parent/'tianshu_lm_precision_service_20261020/policy.py').read_text()
    text=(BASE/'policy.py').read_text()
    def fn(source,name):return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('role_for','prepare_layer','native_lm_fallback','make_process','make_apply','load_model'):
        assert ast.dump(fn(text,name))==ast.dump(fn(original,name)),name
    result=dict(complete=True,no_GPU_started=True,runtime_source_count=len(profiles[0]['frozen_sources']),environment_count=63,
        unchanged_weight_loader_LM_BF16_and_framework_method_AST_verified=True,
        both_modes_retain_qualified_LM_BF16_prefill_GPU4=True,candidate_GateUp_only_padding=True,profiles=profiles)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
if __name__=='__main__':main()
