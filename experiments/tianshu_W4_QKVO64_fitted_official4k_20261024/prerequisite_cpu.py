# SPDX-License-Identifier: Apache-2.0
"""Gate the original N256 run on the completed fresh N128 and both Math runs."""
from pathlib import Path
from candidate_profile import read,digest
from audit import audit_math
BASE=Path(__file__).resolve().parent
SCREEN=BASE.parent/'tianshu_W4_QKVO64_fitted_fresh128_20261024'
PAIR=Path('/workspace/benchmark-results/20261024_W4_QKVO64_fitted_fresh128_pair_tianshu')

def check():
    gate=read(SCREEN/'SCREEN_GATE_32g.json')
    assert gate['complete'] and gate['model_quality_qualified']
    assert gate['meets_1percent_throughput_and_1percent_spread_gate']
    assert gate['paired_total_throughput_change_percent']>=1
    assert gate['exact_model_sources_and_original_commands_verified']
    assert gate['both_arms_equal_additional_weight_payload'] and gate['all1024_performance_requests_succeeded']
    assert (PAIR/'phase.txt').read_text().strip()=='joint_fresh128_complete_pending_review'
    screen_state=read(PAIR/'candidate/result.json')
    assert screen_state['phase']=='complete' and screen_state['evaluation_exit_code']==0
    assert len(screen_state['frozen_sources'])==997
    for batch in (8,64):
        root=Path(f'/workspace/benchmark-results/20261024_W4_QKVO64_fitted_v2_math{batch}_tianshu')
        state=read(root/'result.json')
        assert state['phase']=='complete' and state['evaluation_exit_code']==0
        assert state['frozen_sources']==screen_state['frozen_sources']
        assert state['server_command']==screen_state['server_command']
        q=audit_math(root,state)
        assert q==read(root/'accuracy_audit.json') and q['above95'] and q['total']==105
        assert q['correct']==gate['quality'][str(batch)]['correct']
    return dict(screen_gate_path=str(SCREEN/'SCREEN_GATE_32g.json'),
        screen_gate_sha256=digest(SCREEN/'SCREEN_GATE_32g.json'),
        paired_fresh128_change_percent=gate['paired_total_throughput_change_percent'],
        model_sources=screen_state['frozen_sources'],server_command=screen_state['server_command'],
        quality=gate['quality'],screen_candidate_full_environment=screen_state['full_environment'],
        screen_candidate_root=str(PAIR/'candidate'))
