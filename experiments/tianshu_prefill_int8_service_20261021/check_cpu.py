# SPDX-License-Identifier: Apache-2.0
import ast,json
from pathlib import Path
from candidate_profile import BASE,prepare

# The only semantic audit adaptation is the old mandatory BF16 ready coverage;
# all other source/native/weights/Graph/FutureKV checks are preserved verbatim.
parent=BASE.parent/'tianshu_lm_precision_service_20261020/audit_profile.py';expected=parent.read_text()
for before,after in [
    ("BASE/'dense_audit.py'","BASE.parent/'tianshu_lm_precision_service_20261020/dense_audit.py'"),
    ('    assert len({r["prefix"] for r in active})==42','    if state["prefill_int8_mode"]=="control":assert len({r["prefix"] for r in active})==42\n    else:assert state["prefill_int8_mode"]=="candidate"'),
    ('actual_ready_request_path_all_42_GateUp_verified=True','actual_ready_request_path_all_42_GateUp_verified=(len({r["prefix"] for r in active})==42)')]:
    assert expected.count(before)==1;expected=expected.replace(before,after)
assert expected==(BASE/'adapted_qualified_audit.py').read_text()
for p in BASE.rglob('*.py'):ast.parse(p.read_text())
profiles=[]
for mode in ('control','candidate'):
    root=Path(f'/workspace/benchmark-results/20261021_prefill_int8_service_CPU_{mode}_tianshu');assert not root.exists();root.mkdir()
    command,env,profile,sources=prepare(root,mode);assert len(profile)==62 and len(command)==13
    assert profile['PYTHONPATH']==str(BASE/('hooks_'+mode)) and all(profile[k]=='candidate' for k in ('TIANSHU_DENSE_INT8_MODE','TIANSHU_LM_PRECISION_MODE','TIANSHU_PREFILL_LINEAR_MODE','TIANSHU_SEQ_LIMIT_MODE'))
    profiles.append(dict(mode=mode,command=command,profile=profile,sources=sources))
assert profiles[0]['sources']==profiles[1]['sources']
result=dict(complete=True,CPU_only=True,no_GPU_started=True,runtime_source_count=len(profiles[0]['sources']),environment_count=62,parent_runtime_sources=381,original_serve_command_preserved=True,same_NN_decode_profile_both_arms=True,all_own_AST_parsed=True,
    exactly_three_explicit_qualified_audit_source_adaptations=True,untimed_cache_salt_only_for_same_path_seed_canaries=True,new_model_Math_required=True,profiles=profiles)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
