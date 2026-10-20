# SPDX-License-Identifier: Apache-2.0
"""Validate this intrusive diagnostic's real source/environment and call evidence."""
import json
from pathlib import Path
from candidate_profile import BASE,read,digest,frozen_check

def audit_runtime(root,state,official=False):
    assert state['official_benchmark'] is False and state['adopted'] is False
    assert official is True # inherited runner requests C64 binding checks; it is not an official score
    frozen_check(state['frozen_sources'])
    for path,sha in state['frozen_sources'].items():assert digest(root/'source_snapshot'/Path(path).relative_to('/'))==sha
    observed=read(root/'observed_server_environment.json')
    assert observed['environment']==state['profile_environment'] and len(observed['environment'])==66
    assert observed['command']==state['server_command'] and len(observed['command'])==13
    env=observed['environment']
    assert env['PYTHONPATH']==str(BASE/'hooks') and env['TIANSHU_CPU_PROFILE_PROBE']=='1'
    assert all(env[k]=='candidate' for k in ('TIANSHU_LM_PRECISION_MODE','TIANSHU_DENSE_INT8_MODE','TIANSHU_PREFILL_LINEAR_MODE','TIANSHU_SEQ_LIMIT_MODE'))
    assert state['canary_before']['text']==state['canary_after']['text']
    assert len(state['workloads'])==2 and [w['label'] for w in state['workloads']]==['warmup','measured']
    for wave in state['workloads']:
        assert wave['concurrency']==64 and len(wave['responses'])==64 and wave['input_tokens']==4096 and wave['output_tokens']==256
        for response in wave['responses']:assert response['usage']==dict(prompt_tokens=4096,completion_tokens=256,total_tokens=4352)
    dense=read(next((root/'dense_int8_records').glob('bindings_*.json')))
    signature=sorted([{k:r[k] for k in ('role','prefix','shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes')} for r in dense['banks']],key=lambda r:r['prefix'])
    assert signature==read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')
    assert dense['load_complete'] and dense['actual_packed_banks']==43 and dense['packed_payload_bytes']==1326938112
    assert dense['lm_precision_mode']=='candidate' and dense['supported_M_by_role']['lm_head']==[64] and dense['startup_mode']=='gpu4'
    assert len({r['bank_id'] for r in dense['launches'].values() if r['role']=='gateup' and r['M']==64 and r['actually_inside_capture']})==42
    assert any(r['role']=='lm_head' and r['M']==64 and not r['actually_inside_capture'] and r['calls']>0 for r in dense['launches'].values())
    assert any(r['M']==8 and not r['actually_inside_capture'] for r in dense['BF16_lm_launches'].values())
    scheduler=read(state['scheduler_profile_record']);assert scheduler['complete'] and len(scheduler['frames'])==16
    assert scheduler['installed_after_original_Scheduler_init'] and scheduler['original_schedule_arguments_and_result_preserved']
    assert all(f['requests']==f['total_scheduled_tokens']==64 for f in scheduler['frames'])
    assert len(state['int8_smoke_wave'])==8 and all(r['usage']==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144) for r in state['int8_smoke_wave'])
    profile=read(state['cpu_profile_record']);assert profile['complete'] and len(profile['frames'])==16
    assert not profile['GPU_events_used'] and profile['hook_source_sha256']==digest(BASE/'hooks/sitecustomize.py')
    assert all(frame['dispatch']==dict(mode='FULL',padded_tokens=64) for frame in profile['frames'])
    assert 'No running processes found' in (root/'hardware_after.txt').read_text()
    return dict(complete=True,intrusive_diagnostic_only=True,entire_actual_environment_count=66,all_frozen_sources_unchanged=True,
        actual_C64_INT8_all_42_GateUp_and_LM_verified=True,actual_C8_LM_BF16_verified=True,all_43_packed_banks_match_qualified=True,
        actual_FULL_decode_profile_frames=16,seeded_canary_preserved=True,no_GPU_kernel_time_or_benchmark_claim=True)
