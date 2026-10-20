# SPDX-License-Identifier: Apache-2.0
"""Freeze a fully composed service pair after the real GPU capacity proof."""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
for path in BASE.rglob('*.py'):ast.parse(path.read_text())
component=BASE.parent/'tianshu_small_arena_20261020';gpu=json.loads((component/'GPU_GATE_32g.json').read_text())
assert gpu['complete'] and gpu['independently_rechecked'] and gpu['staging_GPU_checks']==22 and gpu['fallback_branch_checks']==2
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*BASE.rglob('*.py'),BASE/'batch_policy.json',component/'policy.py',component/'CPU_GATE_32g.json',component/'GPU_GATE_32g.json']}
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,sources=sources,GPU_component_gate_sha256=hashlib.sha256((component/'GPU_GATE_32g.json').read_bytes()).hexdigest()),indent=2)+'\n')
sys.path.insert(0,str(BASE));import candidate_profile
root=Path('/workspace/benchmark-results/20261020_small_arena_composition_CPU_tianshu');root.mkdir(exist_ok=False)
profiles=[]
for mode in ('control','candidate'):profiles.append(candidate_profile.prepare(root/mode,mode))
a,b=profiles;assert a[0]==b[0] and a[3]==b[3] and len(a[2])==48
for key,value in a[2].items():
    if key=='TIANSHU_SMALL_ARENA_MODE':assert (value,b[2][key])==('control','candidate')
    else:assert value.replace(str(root/'control'),'<RUN>')==b[2][key].replace(str(root/'candidate'),'<RUN>'),key
# Check that the reused long-context audit still accepts the known current control evidence.
spec=importlib.util.spec_from_file_location('_arena_known_control_audit',BASE/'long_runtime_audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
known=Path('/workspace/benchmark-results/20261020_headroom_service16k_tianshu/control')
state=json.loads((known/'result.json').read_text());checked=audit.audit_runtime(known,dict(state,mode='candidate',arena_mode='control'))
assert checked['actual_FL_Graph_capture_and_replay_verified'] and checked['active_kernel_sha256']=='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
result=dict(complete=True,CPU_only=True,profile_source_count=len(a[3]),profile_environment_count=48,only_mode_difference=True,
            same_original_serve_argv=True,both_modes_S16=True,watermark_unchanged=True,known_native_Graph_audit_rechecked=True,
            frozen_sources=a[3],control_environment=a[2],candidate_environment=b[2])
(BASE/'COMPOSITION_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(complete=True,CPU_only=True,sources=len(a[3]),environment=48)))
