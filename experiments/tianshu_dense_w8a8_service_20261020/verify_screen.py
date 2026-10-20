# SPDX-License-Identifier: Apache-2.0
"""Independently re-audit sources, dispatch, full request accounting and medians."""
import hashlib,json,statistics,sys
from pathlib import Path
from audit_profile import audit_runtime
from candidate_profile import frozen_check
BASE=Path(__file__).resolve().parent
root=Path(sys.argv[1]);runs={m:json.loads((root/m/'result.json').read_text()) for m in ('control','candidate')}
control,candidate=runs['control'],runs['candidate'];assert control['phase']==candidate['phase']=='complete'
assert control['frozen_sources']==candidate['frozen_sources'];frozen_check(control['frozen_sources'])
assert control['server_command']==candidate['server_command'] and len(control['server_command'])==13
normalized={m:{k:v.replace(s['output_root']+'/', '<ROOT>/') for k,v in s['profile_environment'].items()} for m,s in runs.items()}
assert {k for k in normalized['control'] if normalized['control'][k]!=normalized['candidate'][k]}=={'TIANSHU_DENSE_INT8_MODE'}
rounds=[];audits={}
for mode,state in runs.items():
    assert len(state['screen_rounds'])==3 and len(state['int8_smoke_wave'])==8 and len(state['seeded_canaries'])==2
    assert state['seeded_canaries'][0]['text']==state['seeded_canaries'][1]['text']
    audits[mode]=audit_runtime(root/mode,state,official=True);assert audits[mode]==state['profile_audit']
    assert len(state['profile_environment'])==56 and state['profile_environment']['TIANSHU_PAIR_LOAD_MODE']=='candidate'
    for response in state['int8_smoke_wave']:
        assert response['usage']==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144) and response['text']
    for index,row in enumerate(state['screen_rounds']):
        assert row['round']==index+1 and row['successful_requests']==64 and len(row['responses'])==64
        assert row['total_tokens_per_second']==64*5120/row['duration_seconds']
        assert row['output_tokens_per_second']==64*1024/row['duration_seconds']
        assert row['mean_ttft_seconds']==statistics.mean(x['ttft_seconds'] for x in row['responses'])
        for offset,response in enumerate(row['responses']):
            assert response['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120)
            assert response['input_seed']==control['screen_rounds'][index]['responses'][offset]['input_seed']
        rounds.append(dict(mode=mode,round=index+1,included=index>0,total_tok_s=row['total_tokens_per_second'],output_tok_s=row['output_tokens_per_second'],
            mean_TTFT_ms=1000*row['mean_ttft_seconds'],successful_requests=64,new_autotune_records=len(row['tuning_records'])))
assert control['seeded_canaries'][0]['text']==candidate['seeded_canaries'][0]['text']
means={m:statistics.mean(r['total_tok_s'] for r in rounds if r['mode']==m and r['included']) for m in runs}
gain=100*(means['candidate']/means['control']-1)
gate=dict(complete=True,independently_rechecked=True,component_and_service_only=True,official_benchmark=False,adopted=False,fresh_Math_not_run=True,
    input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds_per_arm=3,excluded_rounds=[1],
    total_screen_requests_successful=384,C8_smoke_requests=16,C1_canary_requests=4,failed_requests=0,
    control_total_tok_s=means['control'],candidate_total_tok_s=means['candidate'],gain_percent=gain,
    spread_percent={m:runs[m]['spread_percent'] for m in runs},frozen_source_count=len(control['frozen_sources']),profile_environment_count=56,
    same_sources_and_original_argv=True,only_profile_mode_differs=True,paired_K_always_candidate=True,
    KV_blocks={m:audits[m]['actual_KV_blocks'] for m in runs},extra_weight_bytes={m:audits[m]['extra_packed_weight_bytes'] for m in runs},
    candidate_C8_actual_branch_and_C64_capture_replay_verified=True,LM_head_actual_outside_model_Graph_calls_verified=True,
    all_weight_packing_inside_actual_active_model_profiler_verified=True,
    frozen_sources=control['frozen_sources'],raw_root=str(root),rounds=rounds,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(BASE/'DIAGNOSTIC_4K_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
(BASE/'DIAGNOSTIC_4K_ROUNDS_32g.json').write_text(json.dumps(rounds,indent=2)+'\n')
(root/'independent_screen_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k not in ('frozen_sources','rounds')}))
