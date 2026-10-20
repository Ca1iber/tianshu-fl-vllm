# SPDX-License-Identifier: Apache-2.0
"""Offline verify full packed-load component evidence, references and event statistics."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to('/workspace/benchmark-results')
raw=root/'result.json';data=json.loads(raw.read_text())
assert data['complete'] and data['phase']=='complete' and data['numerical_passed']
assert data['same_Graph_Q_K_V_table_lengths_verified'] and not data['adoption_allowed']
assert not data['official_benchmark'] and not data['model_executed'] and not data['service_executed']
for name,h in data['frozen_sources'].items():
    assert sha(name)==sha(root/'source_snapshot'/Path(name).relative_to('/'))==h,name
assert 'No running processes found' in (root/'hardware_before.txt').read_text()
assert len(data['cases'])==12 and len({c['name'] for c in data['cases']})==12
checks=cpu=0;cases=[]
for row in data['cases']:
    assert row['labels']==['control','k32','v32','kv32'] and len(row['phases'])==2
    numerical=[]
    for phase in row['phases']:
        assert phase['Q_K_V_table_lengths_slots_changed_in_place']==bool(phase['phase'])
        assert len(phase['checks'])==4
        for check in phase['checks']:
            assert check['RNG_unchanged'] and check['all_storage_bitwise_exact'] and check['output_and_segment_guards']
            assert check['physical_page_bytes']==12288 and check['V_max_quantization_abs']==0
            error=check['numerical'];assert error['passed'] and error['failed_elements']==0 and math.isfinite(error['max_abs'])
            assert check['cpu_checks'] and all(c['passed'] and c['failed_elements']==0 for c in check['cpu_checks'])
            checks+=1;cpu+=len(check['cpu_checks']);numerical.append(error['max_abs'])
    assert len(row['raw_ms'])==4 and all(len(samples)==9 for samples in row['raw_ms'])
    medians=[statistics.median(samples) for samples in row['raw_ms']]
    assert all(math.isfinite(x) and x>0 for samples in row['raw_ms'] for x in samples)
    assert medians==row['median_ms']
    speeds=[100*(medians[0]/v-1) for v in medians[1:]]
    assert speeds==row['candidate_speedup_percent']
    cases.append(dict(name=row['name'],batch=row['batch'],strided=row['strided'],labels=row['labels'],median_ms=medians,
                      candidate_speedup_percent=speeds,max_abs_vs_FP32=max(numerical),actual_launch=row['actual_launch']))
base=Path(__file__).resolve().parent
result=dict(complete=True,independently_rechecked=True,adopted=False,official_benchmark=False,
            full_GPU_reference_checks=checks,CPU_reference_vectors=cpu,whole_cache_bytes_and_guards_verified=True,
            same_Graph_two_phases_verified=True,RNG_unchanged=True,
            bitwise_output_equality_to_control_asserted_in_frozen_probe=True,
            interpretation='K32 loses substantially at high batch; V32 below 0.5 percent is not sufficient to run a new service/Math campaign',
            time_scope=data['time_scope'],frozen_sources=data['frozen_sources'],model_sources=data['model_sources'],
            cases=cases,raw_result_path=str(raw),raw_result_sha256=sha(raw),analyzer_source_sha256=sha(__file__))
(base/'GPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
with (base/'COMPONENT_ROUNDS_32g.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['case','variant','median_ms','speedup_percent'])
    for row in cases:
        for i,label in enumerate(row['labels']):writer.writerow([row['name'],label,row['median_ms'][i],0 if not i else row['candidate_speedup_percent'][i-1]])
print(json.dumps(dict(complete=True,full_GPU_reference_checks=checks,CPU_reference_vectors=cpu,cases=len(cases))))
