# SPDX-License-Identifier: Apache-2.0
"""Recompute archived CPU selections and all alternating event timing statistics."""
import hashlib,json,statistics,sys
from pathlib import Path
import torch
from safetensors import safe_open
BASE=Path(__file__).resolve().parent
root=Path(sys.argv[1]).resolve();result=json.loads((root/'result.json').read_text())
assert result['complete'] and result['phase']=='complete' and len(result['cases'])==13
assert not result['adopted'] and not result['official_benchmark'] and result['model_accuracy_not_evaluated']
for path,sha in result['frozen_sources'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((root/'source_snapshot'/Path(path).relative_to('/')).read_bytes()).hexdigest()==sha
with safe_open(result["weight"]["checkpoint"],framework="pt",device="cpu") as checkpoint:
    actual_weight=torch.cat([checkpoint.get_tensor(key) for key in result["weight"]["keys"]],0).contiguous()
assert hashlib.sha256(memoryview(actual_weight.view(torch.uint8).numpy())).hexdigest()==result["weight"]["BF16_bits_sha256"]
checks=0;elements=0;rows=[]
for case in result['cases']:
    assert len(case['phases'])==len(case['phase_files'])==3 and case['timing_input_guards_passed']
    for phase,path in enumerate(case['phase_files']):
        data=torch.load(root/path,map_location='cpu',weights_only=True);proof=case['phases'][phase]
        assert data['phase']==phase and data['M']==case['M'] and proof['phase']==phase
        assert proof['inputs_unchanged'] and proof['input_guards_passed'] and proof['RNG_unchanged']
        assert torch.equal(data["original_weight_rows"],actual_weight[data["columns"]])
        reference=data['input_rows'].float()@(data['original_weight_rows'].float()*data['weight_sign']).T
        for name,output in data['outputs'].items():
            assert proof['checks'][name]['full_GPU_reference_passed'] and bool(torch.isfinite(output).all())
            assert not bool(((output.float()-reference).abs()>(.02+.01*reference.abs())).any())
            checks+=1;elements+=output.numel()
    times=[row for row in result['timing_rounds'] if row['M']==case['M']]
    assert len(times)==9 and sorted(row['round'] for row in times)==list(range(9))
    for row in times:
        assert row['order']==(['mm','linear'] if row['round']%2==0 else ['linear','mm'])
        assert row['replays']==20 and row['mm_ms']>0 and row['linear_ms']>0
    mm=statistics.median(row['mm_ms'] for row in times);linear=statistics.median(row['linear_ms'] for row in times)
    rows.append(dict(M=case['M'],N=case['N'],K=case['K'],mm_ms=mm,linear_ms=linear,linear_gain_percent=(mm/linear-1)*100,
        all_three_phases_bitwise_same=all(p['bitwise_same'] for p in case['phases']),max_cross_method_abs=max(p['cross_method_max_abs'] for p in case['phases']),
        mm_range_percent=(max(t['mm_ms'] for t in times)-min(t['mm_ms'] for t in times))/mm*100,
        linear_range_percent=(max(t['linear_ms'] for t in times)-min(t['linear_ms'] for t in times))/linear*100))
gate=dict(complete=True,independently_CPU_rechecked=True,raw_root=str(root),cases=13,full_GPU_checks_asserted=78,
    independent_CPU_selected_checks=checks,independent_CPU_selected_output_elements=elements,timing_groups=len(result['timing_rounds']),
    model_accuracy_not_evaluated=True,service_not_executed=True,official_benchmark=False,adopted=False,
    offline_reference_scope='Archived selected rows/columns, not a full GPU-output recheck',frozen_sources=result['frozen_sources'],weight=result['weight'],comparisons=rows)
(BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
(BASE/'GPU_ROUNDS_32g.json').write_text(json.dumps(result['timing_rounds'],indent=2)+'\n')
print(json.dumps(gate))
