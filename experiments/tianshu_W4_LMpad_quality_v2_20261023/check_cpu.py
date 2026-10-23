# SPDX-License-Identifier: Apache-2.0
import ast,json,sys
from pathlib import Path
import quality_profile as q
root=Path('/workspace/benchmark-results/20261023_W4_LMpad_quality_v2_CPU_tianshu');assert not root.exists() and not (q.BASE/'CPU_GATE_32g.json').exists();root.mkdir()
for p in q.BASE.glob('*.py'):ast.parse(p.read_text())
prepared={}
for batch in (8,64):
 arm=root/f'C{batch}';arm.mkdir();command,env,profile,sources=q.prepare(arm,q.HOLDOUT,batch=batch);assert len(command)==13 and len(sources)==q.read(q.SERVICE/'CPU_GATE_32g.json')['expected_model_source_count'] and len(profile)==62 and len(profile|{'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']})==63
 prepared[batch]=dict(command=command,profile=q.normalized_environment(profile,arm),sources=sources,controllers=q.read(arm/'controller_sources.json'))
a,b=prepared[8],prepared[64];assert a['sources']==b['sources'] and a['command']==b['command'] and a['controllers']==b['controllers'];assert {k for k in a['profile'] if a['profile'][k]!=b['profile'][k]}=={'TIANSHU_FUTURE_KV_MODE'} and a['profile']['TIANSHU_FUTURE_KV_MODE']=='candidate' and b['profile']['TIANSHU_FUTURE_KV_MODE']=='control'
assert q.down.core.q8.parent is q.adapted_C8 and q.adapted_C8.parent is q.original_C8_parent
assert q.down.core.q64.parent is q.adapted_C64 and q.adapted_C64.parent is q.original_C64_parent and q.down.own is q.adapted_Down
assert q.adapted_C64.parent is q.down.core.aux.dense.parent and q.adapted_C64.parent.audit_runtime is q.down.core.aux.audit_native_admission
for module in (q.adapted_C8,q.adapted_C64):assert Path(module.own.__file__).resolve()==q.SERVICE/'audit_profile.py'
assert q.down.own.__file__.endswith('/tianshu_W4_LMpad_service_v2_20261023/inherited_down_audit.py')
assert 'for m in range(1,65)' in (q.SERVICE/'policy.py').read_text() and "==list(range(1,65))" in (q.SERVICE/'audit_profile.py').read_text()
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,GPU_not_run=True,model_not_started=True,source_count=len(a['sources']),original13argv=True,environment_count=63,C8_C64_same_sources_and_argv=True,C64_only_futureKV_control=True,C64_actual_native_admission_audit_retained=True,C8_old_BF16_guard_explicitly_replaced=True,C64_old_W8_guard_explicitly_replaced=True,Down_stale_LM64_INT8_guard_fixed_before_GPU=True,all64_M_startup_warmup_frozen=True,controllers_before_CPU_gate_and_EXECUTION=a['controllers'],not_model_quality_or_speed_qualification=True)
(q.BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='controllers_before_CPU_gate_and_EXECUTION'}))
