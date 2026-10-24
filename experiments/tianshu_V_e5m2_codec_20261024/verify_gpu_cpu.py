# SPDX-License-Identifier: Apache-2.0
"""Post-run source/guard/record and event-arithmetic audit, without GPU imports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
base=Path(__file__).resolve().parent
assert not (base/'GPU_GATE_32g.json').exists()
root=a.root.resolve()
data=json.loads((root/'result.json').read_text())
assert data['complete'] and data['phase']=='complete'
assert data['full_GPU_FP32_and_selected_CPU_FP64_references_passed']
assert not any(data[k] for k in ('engine_allocator_connected','prefill_executed','model_executed','service_executed','official_benchmark','adopted'))
assert all(v is True for v in data['codec_validation'].values())
for path,sha in data['frozen_sources'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((root/'source_snapshot'/Path(path).relative_to('/')).read_bytes()).hexdigest()==sha
assert data['launch_proof']['normalized_launch_AST_unchanged'] and data['launch_proof']['reducer_AST_unchanged']
assert len(data['cases'])==10
rows=[];gpu_cells=cpu_scalars=0;by_name={}
for case in data['cases']:
    assert case['labels']==['control','BF16_PV','FP16_PV']
    assert len(case['phases'])==2 and len(case['raw_ms'])==3
    by_name[case['name']]=case
    for index,values in enumerate(case['raw_ms']):
        assert len(values)==9 and all(math.isfinite(v) and v>0 for v in values)
        median=statistics.median(values)
        assert math.isclose(median,case['median_ms'][index],rel_tol=1e-12)
        launch=case['actual_launch'][index]
        control=case['actual_launch'][0]
        for key in ('IS_3D','sequences','segments','tile','grid','warps','stages','BLOCK_M','BLOCK_Q','pure_decode'):
            assert launch[key]==control[key],(case['name'],key)
        assert launch['query_dtype']=='torch.bfloat16' and launch['key_dtype']=='torch.uint8'
        assert launch['value_dtype']==('torch.bfloat16' if index==0 else 'torch.uint8')
        assert launch['address_selection']['address_bits']==64
        reduction=100*(1-median/case['median_ms'][0])
        if index:assert math.isclose(reduction,case['candidate_time_reduction_percent'][index-1],abs_tol=1e-10)
        rows.append(dict(case=case['name'],batch=case['batch'],label=case['labels'][index],median_ms=median,
            time_reduction_percent=reduction,trial_spread_percent=100*(max(values)-min(values))/median,
            n_regs=launch['n_regs'],n_spills=launch['n_spills']))
    for phase_id,phase in enumerate(case['phases']):
        assert phase['phase']==phase_id and phase['same_Graph_Q_K_V_table_lengths_slots_changed']==bool(phase_id)
        assert len(phase['checks'])==3
        for index,check in enumerate(phase['checks']):
            assert check['label']==case['labels'][index]
            assert check['numerical']['passed'] and check['numerical']['failed_elements']==0
            assert check['numerical']['atol']==.002 and check['numerical']['rtol']==.01
            assert check['physical_page_bytes']==(12288 if index==0 else 8192)
            assert check['quantized_V_reference']==bool(index)
            for key in ('full_cache_storage_bitwise_exact','output_and_scratch_guards','RNG_unchanged','full_inputs_bitwise_unchanged'):assert check[key]
            gpu_cells+=case['batch']*16*128
            expected_vectors=(1 if case['batch']==1 else 3)*4
            assert len(check['CPU_FP64'])==expected_vectors
            for ref in check['CPU_FP64']:
                assert ref['passed'] and ref['failed_elements']==0 and ref['scalars']==128
                assert ref['atol']==.002 and ref['rtol']==.01
                cpu_scalars+=ref['scalars']
qualified=[]
for index,label in enumerate(('control','BF16_PV','FP16_PV')):
    if index and all(by_name[name]['candidate_time_reduction_percent'][index-1]>=5 for name in ('decode64_4096','decode64_5120')):
        qualified.append(label)
gate=dict(complete=True,CPU_only=True,component_runtime_numerical_records_verified=True,
    full_GPU_FP32_output_cells=gpu_cells,selected_CPU_FP64_scalars=cpu_scalars,
    raw_event_medians_independently_recomputed=True,all_sources_and_snapshots_verified=True,
    all_guards_and_changed_data_replays_verified=True,qualified_5percent_B64_4K_and_5K_candidates=qualified,
    model_or_service_qualified=False,adopted=False,raw_root=str(root),
    result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
    verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    offline_scope='Source snapshots, runtime validation records and raw timing arithmetic; full tensors were not serialized for offline attention recomputation')
with (base/'COMPONENT_KEY_TIMES_32g.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(base/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps(gate,indent=2))
