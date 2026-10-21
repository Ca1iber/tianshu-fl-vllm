# SPDX-License-Identifier: Apache-2.0
"""Archive only completed BF16 range phases; preserve original failed plan."""
import csv,hashlib,json,statistics
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261021_m_range_BF16_GPU_tianshu')
result=json.loads((ROOT/'result.json').read_text())
assert result['phase']=='failed' and not result['complete'] and len(result['cases'])==51
assert "('o', 8, 1," in result['error']
assert not torch.cuda.is_initialized();torch.set_num_threads(4)
for filename,digest in result['frozen_sources'].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
    copied=ROOT/'source_snapshot'/Path(filename).relative_to('/')
    assert hashlib.sha256(copied.read_bytes()).hexdigest()==digest,filename
checks=0;full_online=0;rows=[]
for case in result['cases']:
    assert all(v['passed'] for v in case['full_output_GPU_checks'].values())
    assert all(v['passed'] for v in case['timed_full_output_GPU_checks'].values())
    payload=torch.load(ROOT/f"{case['role']}_M{case['M']}_phase{case['phase']}.pt",map_location='cpu',weights_only=False)
    expected=(payload['A'].double()@payload['W'].double().T).to(torch.bfloat16).float()
    for stage in ('outputs_before_timing','outputs_after_timing'):
        for label,value in payload[stage].items():
            assert torch.isfinite(value).all() and not ((value.float()-expected).abs()>.02+.01*expected.abs()).any()
            checks+=value.numel()
    full_online+=4*case['elements_checked_per_implementation']
    for label,median in case['median_ms'].items():
        rows.append(dict(role=case['role'],M=case['M'],phase=case['phase'],label=label,median_ms=median,
            vs_same_tile_NT_pct=(case['median_ms']['matched_NT']/median-1)*100,
            vs_native_library_pct=(case['median_ms']['native_library']/median-1)*100))
qkv=[case for case in result['cases'] if case['role']=='qkv']
assert len(qkv)==48 and len({case['M'] for case in qkv})==24
for m in {case['M'] for case in qkv}:assert {case['phase'] for case in qkv if case['M']==m}=={0,1}
gate=dict(complete=True,scope='Independent CPU recheck of completed partial evidence only',CPU_only=True,
    original_GPU_plan_passed=False,original_GPU_phase='failed',planned_phases=144,completed_phases=51,
    QKV_24_M_two_phases_complete=True,O_completed_phases=3,Down_not_executed=True,
    saved_actual_selected_elements_independently_recomputed=checks,
    full_elements_GPU_checked_online_only=full_online,
    full_outputs_not_saved_or_CPU_reloaded=True,numerical_tolerances_unchanged=True,
    original_failure=result['error'],source_and_snapshot_unchanged=True,
    raw_result_sha256=hashlib.sha256((ROOT/'result.json').read_bytes()).hexdigest(),
    verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),service_executed=False,adoption_allowed=False)
(BASE/'PARTIAL_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
with (BASE/'PARTIAL_ROUNDS_32g.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(json.dumps(gate,indent=2))
