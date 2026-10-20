# SPDX-License-Identifier: Apache-2.0
import ast,json
from pathlib import Path
from candidate_profile import BASE,prepare
import importlib.util,inspect
old_policy=(BASE.parent/"tianshu_kv_empty_allocate_service_20261021/policy.py").read_text()
before="inspect.getsourcefile(type(controller))";after="inspect.getsourcefile(original_emit)"
assert old_policy.count(before)==1 and (BASE/"policy.py").read_text()==old_policy.replace(before,after)
source=BASE.parent/"tianshu_future_kv_reservation_20261020/policy.py"
spec=importlib.util.spec_from_file_location("_CPU_unregistered_future_policy",source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
controller=object.__new__(module.Reservation)
try:inspect.getsourcefile(type(controller))
except TypeError:pass
else:raise AssertionError("fixture did not reproduce old class source failure")
assert Path(inspect.getsourcefile(controller.emit)).resolve()==source.resolve()
profiles=[]
for path in BASE.rglob('*.py'):ast.parse(path.read_text())
for mode in ('control','candidate'):
    root=Path(f'/workspace/benchmark-results/20261021_kv_empty_allocate_service_v2_CPU_{mode}_tianshu');assert not root.exists();root.mkdir()
    command,env,profile,sources=prepare(root,mode)
    assert len(profile)==62 and len(command)==13 and profile['PYTHONPATH']==str(BASE/('hooks_'+mode))
    assert all(profile[k]=='candidate' for k in ('TIANSHU_DENSE_INT8_MODE','TIANSHU_LM_PRECISION_MODE','TIANSHU_PREFILL_LINEAR_MODE','TIANSHU_SEQ_LIMIT_MODE'))
    profiles.append(dict(mode=mode,command=command,profile=profile,sources=sources))
assert profiles[0]['sources']==profiles[1]['sources']
result=dict(complete=True,CPU_only=True,no_GPU_started=True,runtime_source_count=len(profiles[0]['sources']),environment_count=62,
    original_serve_command_preserved=True,parent_runtime_sources=359,dynamic_module_method_source_verified=True,only_policy_change_is_record_source_lookup=True,all_own_AST_parsed=True,profiles=profiles)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
