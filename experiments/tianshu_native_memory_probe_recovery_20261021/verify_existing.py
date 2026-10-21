# SPDX-License-Identifier: Apache-2.0
"""CPU-only recovery of already saved observations; never rewrites raw state."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261021_native_memory_probe_candidate_tianshu')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state=read(ROOT/'state.json');raw_state_sha=sha(ROOT/'state.json')
assert state['phase']=='failed' and state['error']=='AssertionError()' and state['server_reaped']
assert state['mode']=='candidate' and state['benchmark'] is False and state['model_Math'] is False
sources=state['frozen_sources'];assert len(sources)==426 and len(state['profile_environment'])==62
for path,h in sources.items():
    p=Path(path);assert sha(p)==h
    assert sha(ROOT/'source_snapshot'/p.relative_to('/'))==h
cpu=read(Path('/workspace/vllm-plugin-FL/experiments/tianshu_native_memory_probe_20261021/CPU_GATE_32g.json'))
assert next(r['sources'] for r in cpu['profiles'] if r['mode']=='candidate')==sources
assert {k:state['selected_actual_environment'].get(k) for k in state['profile_environment']}==state['profile_environment']
responses=read(ROOT/'canary_responses.json');assert len(responses)==8
canaries=[]
for r in responses:
    u=r['usage'];assert {k:u[k] for k in ('prompt_tokens','completion_tokens','total_tokens')}==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144)
    assert set(u)=={'prompt_tokens','completion_tokens','total_tokens','prompt_tokens_details'} and u['prompt_tokens_details'] is None
    assert len(r['choices'])==1 and r['choices'][0]['finish_reason']=='length'
    canaries.append(dict(usage=u,text_sha256=hashlib.sha256(r['choices'][0]['text'].encode()).hexdigest()))
files=list((ROOT/'native_memory_probe_records').glob('observed_*.json'));assert len(files)==1
observation=read(files[0]);assert observation['complete'] and observation['mode']=='candidate'
assert sources[observation['observer_source']]==observation['observer_sha256']
b=observation['budget'];expected=b['requested_memory']-b['weights_memory']-b['peak_activation_memory']-b['non_torch_memory']-(b['cudagraph_memory_estimate'] if b['cudagraph_estimate_enabled'] else 0)
assert b['original_KV_memory_bytes']==expected and b['native_budget_equation_exact'] and b['no_extra_profile_run'] and b['no_peak_reset'] and b['native_return_preserved']
configs=[r for p in (ROOT/'static_compile_records').glob('config_*.json') for r in read(p)]
assert configs and all(r['mode']=='candidate' and r['actual_compile_sizes']==[8,64] and r['actual_budget']==4096 and r['actual_max_num_seqs']==64 for r in configs)
files2=list((ROOT/'static_compile_records').glob('backend_*.json'));assert len(files2)==1
backend=read(files2[0]);assert backend['capture_completed'] and backend['after_original_capture'] and backend['actual_compile_sizes']==[8,64]
capture_shapes=sorted({r['runtime_shape'] for r in backend['routes'].values() if r['actually_inside_capture']})
assert 8 in capture_shapes and 64 in capture_shapes
for m in (8,64):
    selected=[r for r in backend['routes'].values() if r['actually_inside_capture'] and r['runtime_shape']==m]
    assert selected and all(r['selected_is_single'] and r['selected_range']==[m,m] and r['compiled'] for r in selected)
assert (ROOT/'server.log').read_text(errors='replace').count('Application startup complete.')==1
assert sha(ROOT/'state.json')==raw_state_sha
result=dict(complete=True,CPU_only_recovery=True,new_GPU_job=False,raw_state_preserved=True,raw_failed_state_sha256=raw_state_sha,
    original_failure='Controller compared complete usage dict to a three-field dict; native extra prompt_tokens_details:null is valid.',
    numerical_source_unchanged=True,runtime_sources=sources,runtime_source_count=426,environment_count=62,
    raw_root=str(ROOT),native_observation=observation,actual_configs=configs,canaries=canaries,actual_capture_shapes=capture_shapes,
    full_C64_request_replay_not_tested=True,model_speedup_not_claimed=True,Math_not_run=True,
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'state.json',ROOT/'server.log',ROOT/'canary_responses.json',*files,*files2,*list((ROOT/'static_compile_records').glob('config_*.json'))]})
(BASE/'RECOVERY_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('complete','CPU_only_recovery','new_GPU_job','raw_state_preserved','runtime_source_count','environment_count','native_observation','actual_capture_shapes')}))
