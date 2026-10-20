# SPDX-License-Identifier: Apache-2.0
"""Offline provenance and arithmetic audit; does not rerun or claim independent GPU execution."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);args=p.parse_args()
root=args.root.resolve();result=json.loads((root/'result.json').read_text());base=Path(__file__).resolve().parent
assert result['complete'] and result['numerical_passed'] and result['same_Graph_Q_K_V_table_lengths_verified']
assert not any(result[k] for k in ('adoption_allowed','engine_allocator_connected','prefill_executed','model_executed','service_executed','official_benchmark'))
for name,expected in result['frozen_sources'].items():
    path=Path(name);assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    snapshot=root/'source_snapshot'/path.relative_to('/')
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==expected,str(snapshot)
assert len(result['cases'])==11
full_gpu=cpu_vectors=0;rows=[]
for case in result['cases']:
    assert case['labels']==['qualified_K8V16','dynamic_K8_INT8_QK']
    assert len(case['phases'])==2 and len(case['actual_launch'])==2
    for phase_id,phase in enumerate(case['phases']):
        assert phase['phase']==phase_id
        assert phase['Q_K_V_table_lengths_slots_changed_in_place']==bool(phase_id)
        assert len(phase['checks'])==2
        for index,check in enumerate(phase['checks']):
            assert check['label']==case['labels'][index]
            assert check['numerical']['passed'] and check['numerical']['failed_elements']==0
            assert all(row['passed'] and row['failed_elements']==0 for row in check['cpu_checks'])
            assert check['all_storage_bitwise_exact'] and check['output_and_segment_guards'] and check['RNG_unchanged']
            assert check['effective_Q_K_from_independent_CPU_codec']==bool(index)
            assert check['physical_page_bytes']==(12416 if index else 12288)
            full_gpu+=1;cpu_vectors+=len(check['cpu_checks'])
    for index,measurements in enumerate(case['raw_ms']):
        assert len(measurements)==9 and all(x>0 for x in measurements)
        assert abs(statistics.median(measurements)-case['median_ms'][index])<1e-12
    control,candidate=case['median_ms'];gain=100*(control/candidate-1)
    assert abs(gain-case['candidate_speedup_percent'][0])<1e-10
    rows.append(dict(name=case['name'],batch=case['batch'],control_ms=control,candidate_ms=candidate,
                     throughput_change_percent=gain,time_increase_percent=100*(candidate/control-1)))
assert full_gpu==44
with (base/'DECODE_ROUNDS_32g.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(base/'DECODE_GATE_32g.json').write_bytes((root/'result.json').read_bytes())
summary=dict(complete=True,official_benchmark=False,adopted=False,full_GPU_reference_checks=full_gpu,
             CPU_selected_vectors=cpu_vectors,cases=len(rows),all_cases_slower=all(x['throughput_change_percent']<0 for x in rows),
             frozen_sources=len(result['frozen_sources']),raw_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
             audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)
(base/'DECODE_AUDIT_32g.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
