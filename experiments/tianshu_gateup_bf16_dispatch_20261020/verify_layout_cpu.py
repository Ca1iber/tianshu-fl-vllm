# SPDX-License-Identifier: Apache-2.0
"""Recheck selections and violating pairs; never convert a numerical failure into a pass."""
import hashlib,json,statistics,sys
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent;root=Path(sys.argv[1]).resolve();result=json.loads((root/'result.json').read_text())
assert result['complete'] and result['phase']=='complete' and len(result['cases'])==6
for path,sha in result['frozen_sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha and hashlib.sha256((root/'source_snapshot'/Path(path).relative_to('/')).read_bytes()).hexdigest()==sha
cases=[];failures=[];checks=0
for case in result['cases']:
    assert len(case['phase_files'])==len(case['phases'])==3
    for phase,path in enumerate(case['phase_files']):
        data=torch.load(root/path,map_location='cpu',weights_only=True);proof=case['phases'][phase]
        assert data['phase']==phase and data['case_id']==case['case_id'] and all(proof[k] for k in ('inputs_bitwise_unchanged','guards_passed','RNG_unchanged'))
        reference=data['input_rows'].float()@data['weight_rows'].float().T
        for name,out in data['outputs'].items():
            cpu_pass=bool(torch.isfinite(out).all()) and not bool(((out.float()-reference).abs()>(.02+.01*reference.abs())).any())
            if proof['checks'][name]['full_GPU_reference_passed']:assert cpu_pass
            else:
                f=data['failures'][name];cpu=(f['input_rows'].float()*f['weight_rows'].float()).sum(1)
                assert torch.equal(cpu,f['CPU_reference']) and not bool(((cpu-f['GPU_reference']).abs()>(.001+.001*cpu.abs())).any())
                actual=f['actual'].float();assert bool(((~torch.isfinite(actual))|((actual-cpu).abs()>(.02+.01*cpu.abs()))).any())
                failures.append(dict(case_id=case['case_id'],phase=phase,method=name,full_GPU_violating_elements=proof['checks'][name]['violating_elements'],independent_CPU_violating_pairs=len(actual)))
            checks+=1
    row=dict(case_id=case['case_id'],M=case['M'],strided_A=case['strided_A'],methods={name:all(p['checks'][name]['full_GPU_reference_passed'] for p in case['phases']) for name in ('mm','linear')})
    times=[t for t in result['timing_rounds'] if t['case_id']==case['case_id']]
    if case['both_methods_passed_all_phases']:
        assert len(times)==9 and all(t['order']==(['mm','linear'] if t['round']%2==0 else ['linear','mm']) and t['replays']==20 for t in times)
        mm=statistics.median(t['mm_ms'] for t in times);linear=statistics.median(t['linear_ms'] for t in times);row.update(mm_ms=mm,linear_ms=linear,linear_gain_percent=(mm/linear-1)*100)
    else:assert not times
    cases.append(row)
gate=dict(complete=True,independently_CPU_rechecked=True,case_count=6,CPU_selected_and_failure_checks=checks,failures=failures,cases=cases,
    tolerance_unchanged=True,failed_methods_not_timed=True,all_contiguous_linear_passed=all(c['methods']['linear'] for c in cases if not c['strided_A']),
    model_executed=False,adopted=False,official_benchmark=False,offline_reference_scope=result['offline_CPU_scope'],frozen_sources=result['frozen_sources'],raw_root=str(root))
(BASE/'LAYOUT_DIAGNOSTIC_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'LAYOUT_DIAGNOSTIC_ROUNDS_32g.json').write_text(json.dumps(result['timing_rounds'],indent=2)+'\n')
print(json.dumps(gate))
