# SPDX-License-Identifier: Apache-2.0
"""Recheck actual saved selected BF16 stage outputs, with explicit scope."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
root=args.root.resolve();result=json.loads((root/'result.json').read_text())
assert result['complete'] and result['phase']=='complete'
for filename,digest in result['frozen_sources'].items():assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
import torch
assert not torch.cuda.is_initialized();torch.set_num_threads(4)
assert len(result['cases'])==len(result['roles'])*len(result['rows'])*2
checks=0;rows=[]
for case in result['cases']:
    payload=torch.load(root/f"{case['role']}_M{case['M']}_phase{case['phase']}.pt",map_location='cpu',weights_only=False)
    expected=(payload['A'].double()@payload['W'].double().T).to(torch.bfloat16).float()
    for stage in ('outputs_before_timing','outputs_after_timing'):
        for label,value in payload[stage].items():
            delta=(value.float()-expected).abs()
            assert torch.isfinite(value).all() and not (delta>.02+.01*expected.abs()).any(),(case['role'],case['M'],case['phase'],stage,label)
            checks+=value.numel()
    assert all(v['passed'] for v in case['full_output_GPU_checks'].values())
    assert all(v['passed'] for v in case['timed_full_output_GPU_checks'].values())
    med=case['median_ms']
    for label,value in med.items():
        rows.append(dict(role=case['role'],M=case['M'],phase=case['phase'],label=label,median_ms=value,
            vs_matched_nt_pct=(med['matched_NT']/value-1)*100,vs_native_library_pct=(med['native_library']/value-1)*100))
summary=dict(complete=True,CPU_only=True,service_executed=False,adoption_allowed=False,
    cases=len(result['cases']),independent_CPU_selected_elements=checks,
    full_GPU_checks_scope='Online complete outputs in frozen GPU driver; CPU reload covers selected saved actual outputs only',
    raw_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),rows=rows)
base=Path(__file__).resolve().parent
(base/'BF16_STAGE_GATE_32g.json').write_text(json.dumps(summary,indent=2)+'\n')
import csv
with (base/'BF16_STAGE_ROUNDS_32g.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({key:value for key,value in summary.items() if key!='rows'},indent=2))
