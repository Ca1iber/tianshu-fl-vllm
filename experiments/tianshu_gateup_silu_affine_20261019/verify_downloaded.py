"""Check frozen sources, phase samples and medians; retain failed candidates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();d=json.loads((root/'result.json').read_text())
assert d['complete'] and not d['official_benchmark'] and not d['adoption_allowed']
for path,digest in d['frozen_sources'].items():
    assert hashlib.sha256((root/'source_snapshot'/path.lstrip('/')).read_bytes()).hexdigest()==digest,path
cases=d['cases'];assert len(cases)==28
passed=[];failed=[]
def check(row):
    assert row['passed'] and row['failed_elements']==0
    assert math.isfinite(row['max_abs']) and row['atol'] in (.001,.1) and row['rtol']==.02
for row in cases:
    check(row['baseline_vs_fp32']);assert row['baseline_tail_preserved']
    for v in row['cpu_reference_checks']:check(v)
    if 'compile_or_runtime_error' in row:
        assert 'raw_ms' not in row
        failed.append(dict(m=row['m'],intermediate=row['intermediate'],configuration=row['configuration'],error=row['compile_or_runtime_error']))
        continue
    assert row['numerical_and_changed_graph_passed'] and row['output_tail_preserved']
    check(row['candidate_vs_fp32']);check(row['candidate_vs_native'])
    actual=row['actual_compiled'];cfg=row['configuration']
    assert actual['warps']==cfg['warps'] and actual['stages']==cfg['stages'] and actual['kernel_hash']
    phase=row['changed_graph_phase']
    for key in ('candidate_vs_fp32','native_vs_fp32','candidate_vs_native'):check(phase[key])
    assert phase['native_tail_preserved'] and phase['candidate_tail_preserved'] and len(phase['cpu_points'])==9
    for point in phase['cpu_points']:
        expected=point['expected'];atol=phase['candidate_vs_fp32']['atol']
        for key in ('candidate','native'):
            assert math.isfinite(point[key]) and abs(point[key]-expected)<=atol+.02*abs(expected)
    raw=row['raw_ms'];assert len(raw)==2 and all(len(v)==7 for v in raw)
    assert all(math.isfinite(v) and v>0 for series in raw for v in series)
    before,after=map(statistics.median,raw)
    assert math.isclose(before,row['baseline_ms'],rel_tol=1e-12)
    assert math.isclose(after,row['candidate_ms'],rel_tol=1e-12)
    assert math.isclose(100*(after/before-1),row['time_change_percent'],abs_tol=1e-10)
    passed.append(dict(m=row['m'],intermediate=row['intermediate'],x_scale=row['x_scale'],weight_scale=row['weight_scale'],configuration=cfg,baseline_ms=before,candidate_ms=after,time_change_percent=row['time_change_percent']))
result=dict(complete=True,component_only=True,source_files=len(d['frozen_sources']),cases=len(cases),passed_cases=len(passed),failed_cases=len(failed),passing_results=passed,failures=failed,full_gpu_tensors_not_reconstructed_offline=True,new_service_and_math_not_run=True,adopted=False)
(root/'local_download_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
