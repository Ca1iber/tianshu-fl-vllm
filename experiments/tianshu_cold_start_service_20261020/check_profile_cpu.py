# SPDX-License-Identifier: Apache-2.0
import ast,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
from candidate_profile import prepare,read

def body(path,class_name=None):
    tree=ast.parse(Path(path).read_text());nodes=tree.body
    if class_name:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==class_name).body
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='build_sql_model_by_db')
    return ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False)
assert body(BASE/'sql_hook.py')==body(BASE.parent/'tianshu_sql_target_reflection_20261020/candidate_sql.py','SQLPersistantModel')
records=[]
for mode in ('native','cpu4','gpu4'):
    root=Path('/workspace/benchmark-results')/f'20261020_cold_start_CPU_gate_{mode}_tianshu';assert not root.exists();root.mkdir()
    command,environment,profile,sources=prepare(root,mode)
    assert len(profile)==59 and profile['TIANSHU_COLD_START_MODE']==mode and profile['VLLM_DISABLE_COMPILE_CACHE']=='1'
    assert len(command)==13 and profile['TIANSHU_DENSE_INT8_MODE']=='candidate'
    records.append(dict(mode=mode,command=command,profile_environment=profile,frozen_sources=sources))
assert records[0]['frozen_sources']==records[1]['frozen_sources']==records[2]['frozen_sources']
gate=dict(complete=True,no_GPU_or_server_started=True,validated_target_reflection_AST_exact=True,runtime_source_count=len(records[0]['frozen_sources']),environment_count=59,
    original_W8_active_in_all_modes=True,all_three_source_maps_identical=True,profiles=records)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({key:value for key,value in gate.items() if key!='profiles'}))
