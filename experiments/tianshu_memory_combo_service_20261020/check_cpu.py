# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
for p in BASE.rglob('*.py'):ast.parse(p.read_text())
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*BASE.rglob('*.py'),BASE/'batch_policy.json',BASE.parent/'tianshu_seq_limit_default_20261020/policy.py',BASE.parent/'tianshu_seq_limit_default_20261020/CPU_GATE_32g.json']}
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,sources=sources),indent=2)+'\n')
sys.path.insert(0,str(BASE));import candidate_profile
root=Path('/workspace/benchmark-results/20261020_memory_combo_composition_CPU_tianshu');root.mkdir(exist_ok=False)
a,b=[candidate_profile.prepare(root/mode,mode) for mode in ('control','candidate')]
assert a[0]==b[0] and a[3]==b[3] and len(a[2])==len(b[2])==50
for key,value in a[2].items():
    if key in ('TIANSHU_SMALL_ARENA_MODE','TIANSHU_SEQ_LIMIT_MODE'):assert (value,b[2][key])==('control','candidate')
    else:assert value.replace(str(root/'control'),'<RUN>')==b[2][key].replace(str(root/'candidate'),'<RUN>'),key
spec=importlib.util.spec_from_file_location('_combo_known_base_runtime',BASE/'long_runtime_audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
known=Path('/workspace/benchmark-results/20261020_small_arena_service4k_tianshu/control');state=json.loads((known/'result.json').read_text())
checked=audit.audit_runtime(known,dict(state,mode='candidate',arena_mode='control',seq_limit=256))
assert checked['actual_FL_Graph_capture_and_replay_verified'] and checked['budget']==4096
result=dict(complete=True,CPU_only=True,GPU_executed=False,profile_source_count=len(a[3]),profile_environment_count=50,
            source_and_original_argv_equal=True,allowed_mode_differences=['TIANSHU_SMALL_ARENA_MODE','TIANSHU_SEQ_LIMIT_MODE'],
            both_modes_S16=True,watermark_unchanged=True,known_base_runtime_rechecked=True,frozen_sources=a[3],
            control_environment=a[2],candidate_environment=b[2],scope='Two-knob memory candidate, no single-knob attribution or performance claim')
(BASE/'COMPOSITION_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(complete=True,CPU_only=True,sources=len(a[3]),environment=50)))
