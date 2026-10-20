# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import policy
from candidate_profile import prepare
BASE=Path(__file__).resolve().parent
root=Path('/workspace/benchmark-results/20261020_dense_w8a8_service_PROFILE_CPU_v4_tianshu');assert not root.exists();root.mkdir()
assert policy.role_for('linear',12288,2048)=='gateup' and policy.role_for('lm_head',130560,2048)=='lm_head'
assert policy.role_for('embedding',130560,2048) is None
for n,k in ((2560,2048),(2048,2048),(2048,6144),(12288,2049),(130560,2049)):
    assert policy.role_for('linear',n,k) is None
for role in ('gateup','lm_head'):
    for m in range(1,130):assert (policy.config_for(role,m) is not None)==(m in (8,64))
profiles={};sources={};commands={}
for mode in ('control','candidate'):
    target=root/mode;target.mkdir();command,environment,profile,frozen=prepare(target,mode)
    assert len(profile)==57 and len(command)==13 and environment['TIANSHU_DENSE_INT8_MODE']==mode
    profiles[mode]={k:v.replace(str(target)+'/', '<ROOT>/') for k,v in profile.items()};sources[mode]=frozen;commands[mode]=command
assert sources['control']==sources['candidate'] and commands['control']==commands['candidate']
assert {k for k in profiles['control'] if profiles['control'][k]!=profiles['candidate'][k]}=={'TIANSHU_DENSE_INT8_MODE'}
gate=dict(complete=True,CPU_only=True,GPU_not_executed=True,embedding_same_shape_excluded=True,other_projections_and_M_excluded=True,
    native_active_memory_profiler_will_be_asserted_in_actual_worker=True,selected_M=[8,64],frozen_source_count=len(sources['control']),
    profile_environment_count=57,source_and_original_argv_equal_between_arms=True,mode_only_environment_difference=True,
    frozen_sources=sources['control'],server_command=commands['control'],raw_root=str(root))
(BASE/'PROFILE_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='frozen_sources'}))
