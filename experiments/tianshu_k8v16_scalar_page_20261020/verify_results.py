# SPDX-License-Identifier: Apache-2.0
import hashlib,json,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261020_k8v16_scalar_page_v2_tianshu')
p=ROOT/'result.json'
x=json.loads(p.read_text())
assert x['complete'] and x['numerical_passed'] and len(x['cases'])==10
for name,expected in x['frozen_sources'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    assert hashlib.sha256((ROOT/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==expected,name
full,cpu=0,0
for case in x['cases']:
    assert len(case['phases'])==2 and len(case['labels'])==2
    for phase in case['phases']:
        assert len(phase['checks'])==2
        for row in phase['checks']:
            assert row['numerical']['passed'] and row['numerical']['failed_elements']==0
            assert row['all_storage_bitwise_exact'] and row['output_and_segment_guards'] and row['RNG_unchanged']
            assert row['physical_page_bytes']==12288
            assert all(c['passed'] and c['failed_elements']==0 for c in row['cpu_checks'])
            full+=1;cpu+=len(row['cpu_checks'])
    for raw,median in zip(case['raw_ms'],case['median_ms'],strict=True):
        assert len(raw)==9 and all(t>0 for t in raw)
        assert statistics.median(raw)==median
    for value,reported in zip(case['median_ms'][1:],case['candidate_speedup_percent'],strict=True):
        assert abs(100*(case['median_ms'][0]/value-1)-reported)<1e-10
result=dict(x)
result.update(component_only=True,adopted=False,full_GPU_FP32_checks=full,CPU_selected_vector_checks=cpu,
              independent_record_audit=True,original_result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
              analyzer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
# Retain each independently checked error statistic; omit repeated CPU vector statistics.
for case in result['cases']:
    for phase in case['phases']:
        for row in phase['checks']:
            checks=row.pop('cpu_checks')
            row['CPU_count']=len(checks)
            row['CPU_all_passed']=True
            row['CPU_max_abs']=max(c['max_abs'] for c in checks)
(BASE/'DECODE_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(complete=True,full_GPU_FP32_checks=full,CPU_selected_vector_checks=cpu)))
