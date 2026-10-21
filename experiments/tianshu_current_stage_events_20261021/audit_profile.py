# SPDX-License-Identifier: Apache-2.0
"""Diagnostic evidence, not a replacement for complete qualification audits."""
import re
from pathlib import Path
from candidate_profile import BASE,read,digest,frozen_check

def audit_runtime(root,state,official=False):
 assert official and state['official_benchmark'] is False and state['adopted'] is False
 frozen_check(state['frozen_sources'])
 original=read(BASE.parent/'tianshu_m_range_prefill_official_results_20261021/OFFICIAL_GATE_32g.json')['model_frozen_sources'];assert len(original)==471 and all(state['frozen_sources'][p]==h for p,h in original.items())
 for p,h in state['frozen_sources'].items():assert digest(root/'source_snapshot'/Path(p).relative_to('/'))==h
 observed=read(root/'observed_server_environment.json');assert observed['environment']==state['profile_environment'] and len(observed['environment'])==66 and observed['command']==state['server_command']
 assert observed['environment']['PYTHONPATH']==str(BASE/'hooks_candidate') and observed['environment']['TIANSHU_STAGE_PROFILE_MODE']=='events' and len(state['server_command'])==13
 assert state['canary_before']['text']==state['canary_after']['text'] and len(state['int8_smoke_wave'])==8
 for wave in state['workloads']:
  assert wave['input_tokens']==4096 and wave['output_tokens']==512 and wave['concurrency']==64 and len(wave['responses'])==64
  assert all(r['usage']==dict(prompt_tokens=4096,completion_tokens=512,total_tokens=4608) for r in wave['responses'])
 dense=read(next((root/'dense_int8_records').glob('bindings_*.json')))
 signature=sorted([{k:r[k] for k in ('role','prefix','shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes')} for r in dense['banks']],key=lambda r:r['prefix'])
 assert signature==read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json') and dense['actual_packed_banks']==43 and dense['packed_payload_bytes']==1326938112
 assert dense['lm_precision_mode']=='candidate' and dense['supported_M_by_role']['lm_head']==[64]
 assert any(r['role']=='lm_head' and r['M']==64 and not r['actually_inside_capture'] for r in dense['launches'].values()) and any(r['M']==8 and not r['actually_inside_capture'] for r in dense['BF16_lm_launches'].values())
 fusion=read(next((root/'nn_silu_fx_records').glob('route_*.json')));captured=[r for r in fusion['routes'].values() if r['M']==64 and r['actually_inside_capture']]
 assert fusion['completed'] and len({r['prefix'] for r in captured})==42 and all(r['component_binary_exact_match'] for r in captured)
 range_record=read(next((root/'m_range_prefill_records').glob('route_*.json')));assert range_record['completed'] and range_record['active']
 assert len({r['prefix'] for r in range_record['routes'].values() if r['stage']=='active' and not r['capturing'] and r['implementation']=='M_range_two_dot_fused' and r['M']==4096})==42
 trace=read(next((root/'graph_segments_records').glob('trace_*.json')));assert any(g['replay_calls']>=100 and g['capture_calls']>=1 and g['frames'].get('64',{}).get('max_computed',0)>=4096 for g in trace['graphs'].values())
 stages=read(state['stage_record']);assert stages['complete'] and stages['frame_count']==40 and not stages['events_inside_graph'] and stages['boundary_synchronizations']==2 and stages['ixsys_not_used']
 assert stages['hook_source_sha256']==digest(BASE/'policy.py') and stages['runner_source_sha256']==state['frozen_sources']['/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py']
 for f in stages['frames']:
  assert f['dispatch']==dict(mode='FULL',padded_tokens=64) and f['context']['requests']==64 and f['context']['min_computed']>=4096
  assert all(x>0 for x in f['gpu_stream_intervals_ms'].values())
 for name in ('before_metrics.txt','after_metrics.txt'):
  values=re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$',(root/name).read_text(),re.MULTILINE);assert values and all(float(v)==0 for v in values)
 assert 'No running processes found' in (root/'hardware_after.txt').read_text()
 return dict(complete=True,diagnostic_only=True,not_full_qualification_replacement=True,original471_exact_subset=True,actual_environment_count=66,all43_weight_signatures_preserved=True,actual_M64_LM_INT8_and_M8_LM_BF16_verified=True,actual_M64_fused42_capture_and_request_graph_replay=True,actual_M4096_prefill_fused42=True,actual_FULL_C64_event_frames=40,seed_canary_preserved=True,zero_preemptions=True,no_official_throughput_claim=True)
