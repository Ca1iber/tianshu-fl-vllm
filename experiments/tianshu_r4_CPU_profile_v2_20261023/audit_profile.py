# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
from candidate_profile import BASE,EXP,read,digest,frozen_check
s=importlib.util.spec_from_file_location('_r4_CPU_real_LM_memory_checks',EXP/'tianshu_W4_LMpad_service_v2_20261023/audit_profile.py');own=importlib.util.module_from_spec(s);s.loader.exec_module(own)

def audit_runtime(root,state,official=False):
 assert official and not state['official_benchmark'] and not state['adopted'];frozen_check(state['frozen_sources'])
 for p,h in state['frozen_sources'].items():assert digest(root/'source_snapshot'/Path(p).relative_to('/'))==h
 observed=read(root/'observed_full_server_environment.json');assert observed['environment']==state['profile_environment']|state['additional_environment'] and len(observed['environment'])==67 and len(observed['command'])==13
 manifest=read(EXP/'tianshu_W4_Down64_LM1to64_release_20261023/RELEASE_MANIFEST_32g.json');assert observed['command']==manifest['server_command']
 baseline={k:v.replace(manifest['raw_root'],str(root)) for k,v in manifest['original_profile_environment'].items()};assert {k for k in baseline if baseline[k]!=observed['environment'][k]}=={'PYTHONPATH'} and set(observed['environment'])-set(baseline)=={'TIANSHU_CPU_PROFILE_PROBE','TIANSHU_CPU_PROFILE_MODE','TIANSHU_CPU_PROFILE_RECORD_DIR','TIANSHU_CPU_PROFILE_ARM_FILE'}
 assert state['canary_before']['text']==state['canary_after']['text'] and len(state['int8_smoke_wave'])==8 and len(state['workloads'])==2
 for wave in state['workloads']:
  assert wave['input_tokens']==4096 and wave['output_tokens']==256 and wave['concurrency']==len(wave['responses'])==64
  assert all(r['usage']==dict(prompt_tokens=4096,completion_tokens=256,total_tokens=4352) for r in wave['responses'])
 lm=own.audit_runtime(root,state,minimum_calls=100,batch=64);assert lm['new_LM64_W4_actual_runtime_calls']>=100 and lm['new_LM8_W4_actual_runtime_calls']>=1 and lm['old_LM1_64_BF16_W8_calls_absent']
 dense=read(next((root/'dense_int8_records').glob('bindings_*.json')));signature=sorted([{k:r[k] for k in ('role','prefix','shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes')} for r in dense['banks']],key=lambda r:r['prefix']);assert signature==read(EXP/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json') and dense['actual_packed_banks']==43
 fusion=read(next((root/'nn_silu_fx_records').glob('route_*.json')));captured=[r for r in fusion['routes'].values() if r['M']==64 and r['actually_inside_capture']];assert len({r['prefix'] for r in captured})==42 and all(r['component_binary_exact_match'] for r in captured)
 trace=read(next((root/'graph_segments_records').glob('trace_*.json')));assert all(trace['graphs'][r['graph_key']]['replay_calls']>=100 for r in captured)
 down=read(next((root/'dots_Down64_records').glob('route_*.json')));assert down['complete'] and down['own_buffers_in_actual_model_accounting'] and len(down['banks'])==42 and len({r['bank_id'] for r in down['routes'].values() if r['phase']=='capture' and r['capturing']})==42
 cpu=read(state['cpu_profile_record']);scheduler=read(state['scheduler_profile_record']);assert cpu['complete'] and scheduler['complete'] and len(cpu['frames'])==len(scheduler['frames'])==16 and cpu['hook_source_sha256']==digest(BASE/'hooks/sitecustomize.py') and not cpu['GPU_events_used']
 assert all(f['dispatch']==dict(mode='FULL',padded_tokens=64) for f in cpu['frames']) and all(f['requests']==f['total_scheduled_tokens']==64 for f in scheduler['frames'])
 assert scheduler['installed_after_original_Scheduler_init'] and scheduler['original_schedule_arguments_and_result_preserved'];assert 'No running processes found' in (root/'hardware_after.txt').read_text()
 return dict(complete=True,intrusive_diagnostic_only=True,original944_model_sources_preserved=True,actual_original13argv=True,actual67_environment_checked=True,only_inference_profile_change='4 CPU profiler flags and composed PYTHONPATH',actual_LM64_W4_calls=lm['new_LM64_W4_actual_runtime_calls'],actual_GU64_fused42_and_model_replays_verified=True,actual_Down64_W4_captures42=True,all43_W8_banks_match_r4=True,all128_requests_usage_checked=True,seeded_canary_preserved=True,no_benchmark_or_GPU_time_claim=True)
