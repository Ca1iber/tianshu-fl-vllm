# SPDX-License-Identifier: Apache-2.0
"""Audit two completed cases plus original terminal failure; never promotes partial runs."""
import argparse,csv,hashlib,json,math,statistics,subprocess
from pathlib import Path
import numpy as np
import domain_cpu
from launch_sweep import CONFIGS
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();base=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
data=json.loads((root/'result.json').read_text())
assert not data['complete'] and data['phase']=='failed'
assert 'decode64_ragged_strided' in data['error'] and 'V8_INT8_S4_T64_W2' in data['error']
assert "'failed_elements': 3" in data['error'] and "'atol': 0.002" in data['error'] and "'rtol': 0.01" in data['error']
assert not any(data[k] for k in ('engine_allocator_connected','prefill_executed','model_executed','service_executed','official_benchmark','adopted'))
assert data['declared_grid']==CONFIGS and all(v is True for v in data['codec_validation'].values())
for path,h in data['frozen_sources'].items():
    assert sha(path)==h and sha(root/'source_snapshot'/Path(path).relative_to('/'))==h
assert data['launch_proof']['normalized_launch_AST_unchanged'] and data['launch_proof']['reducer_AST_unchanged']
for proof in data['source_proofs'].values():
    assert proof['all_edits_inverse_restore_exact_parent'] and proof['native_QK_and_online_softmax_unchanged']
    assert proof['public_API_and_reducer_AST_unchanged'] and proof['current_unpaired_K_load_preserved']
primitive_counts=dict(quantized_elements=0,integer_outputs=0,FP32_output_words=0)
assert {(r['T'],r['warps'],r['phase']) for r in data['primitive_records']}=={(t,w,p) for t,w in ((32,1),(32,2),(64,2)) for p in range(2)}
for record in data['primitive_records']:
    path=Path(record['file']);assert path.is_relative_to(root) and sha(path)==record['sha256']
    with np.load(path,allow_pickle=False) as arrays:counts=domain_cpu.verify_primitive(arrays)
    for key,count in counts.items():assert record[key]==count;primitive_counts[key]+=count
labels=['control']+[cfg['label'] for cfg in CONFIGS];bylabel={cfg['label']:cfg for cfg in CONFIGS}
assert set(data['compiled_kernel_records'])==set(labels)
compiled_files={}
for label,records in data['compiled_kernel_records'].items():
    assert records
    for record in records:
        assert record['recorded_before_timed_Graph_replay'] and record['target']['backend']=='corex'
        assert record['target']['warp_size']==64 and any(path.endswith('.asm') for path in record['files'])
        for path,h in record['files'].items():
            assert Path(path).is_relative_to(root/'codegen') and sha(path)==h;compiled_files[path]=h
assert [case['name'] for case in data['cases']]==['decode64_4096','decode64_5120']
rows=[];gpu_cells=cpu_scalars=0;seen_writer={}
for case in data['cases']:
    assert case['labels']==labels and case['batch']==64 and len(case['phases'])==2
    for index,label in enumerate(labels):
        cfg=dict(segments=4,TILE_SIZE=16,num_warps=1) if index==0 else bylabel[label]
        raw=case['raw_ms'][index];assert len(raw)==9 and all(math.isfinite(t) and t>0 for t in raw)
        median=statistics.median(raw);assert math.isclose(median,case['median_ms'][index],rel_tol=1e-12)
        actual=case['actual_launch'][index]
        assert actual['IS_3D'] and actual['sequences']==64 and actual['BLOCK_M']==16 and actual['BLOCK_Q']==2
        assert (actual['segments'],actual['tile'],actual['warps'],actual['stages'])==(4,cfg['TILE_SIZE'],cfg['num_warps'],1)
        assert actual['query_dtype']=='torch.bfloat16' and actual['key_dtype']=='torch.uint8'
        assert actual['value_dtype']==('torch.bfloat16' if index==0 else 'torch.int8')
        assert actual['address_selection']['address_bits']==64
        if index:assert actual['PV_INT8']==cfg['PV_INT8'] and actual['KV_QUANT_MODE']==2
        reduction=100*(1-median/case['median_ms'][0])
        if index:assert math.isclose(reduction,case['candidate_time_reduction_percent'][index-1],abs_tol=1e-10)
        layout=case['layouts'][index];page_bytes=12288 if index==0 else 8448
        assert layout['page_bytes']==layout['actual_spec_page_bytes']==page_bytes
        assert layout['all_planes_alias_actual_raw'] and layout['all_scale_bytes_in_native_page_accounting']
        assert layout['full_engine_allocator_not_installed']
        rows.append(dict(case=case['name'],label=label,median_ms=median,time_reduction_percent=reduction,
            n_regs=actual['n_regs'],n_spills=actual['n_spills'],tile=cfg['TILE_SIZE'],warps=cfg['num_warps']))
    for phase_id,phase in enumerate(case['phases']):
        assert phase['phase']==phase_id and phase['same_Graph_Q_K_V_scales_table_lengths_slots_changed']==bool(phase_id)
        assert len(phase['checks'])==len(labels)
        for index,check in enumerate(phase['checks']):
            assert check['label']==labels[index] and check['quantized_V_reference']==bool(index)
            assert check['physical_page_bytes']==(12288 if index==0 else 8448)
            assert check['numerical']['passed'] and check['numerical']['failed_elements']==0
            assert check['numerical']['atol']==.002 and check['numerical']['rtol']==.01
            for key in ('full_cache_storage_bitwise_exact','output_and_scratch_guards','RNG_unchanged','full_inputs_bitwise_unchanged'):assert check[key]
            gpu_cells+=64*16*128
            assert len(check['CPU_FP64'])==12
            for c in check['CPU_FP64']:
                assert c['passed'] and c['failed_elements']==0 and c['scalars']==128
                assert c['atol']==.002 and c['rtol']==.01;cpu_scalars+=128
            if index:seen_writer[check['writer_fixture']['file']]=check['writer_fixture']['sha256']
writer_counts=dict(V_integer_elements=0,V_scale_words=0,K_scale_words=0)
writer_paths=sorted(root.glob('writer_*.npz'));assert len(writer_paths)==24
for path in writer_paths:
    if str(path) in seen_writer:assert sha(path)==seen_writer[str(path)]
    with np.load(path,allow_pickle=False) as arrays:counts=domain_cpu.verify_writer(arrays)
    for key,count in counts.items():writer_counts[key]+=count
assert all(row['time_reduction_percent']<0 for row in rows if row['label']!='control')
with (base/'PARTIAL_KEY_TIMES_32g.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
gate=dict(complete=True,CPU_only=True,original_run_completed=False,original_run_failed=True,complete_cases_verified=2,
    all33_sources_and_snapshots_and_compiled_IR_ASM_verified=True,full_GPU_FP32_cells_in_completed_cases=gpu_cells,
    selected_CPU_FP64_scalars_in_completed_cases=cpu_scalars,independent_primitive_counts=primitive_counts,
    independent_actual_writer_counts=writer_counts,writer_fixtures_verified=len(writer_paths),
    strict_numeric_failure_record_preserved=True,failed_output_arrays_not_saved=True,
    failure_count_scope='Original runtime assertion/log only; failed ragged output arrays unavailable for independent recount',
    qualified_fixed_configs=[],model_or_service_qualified=False,adopted=False,raw_root=str(root),
    result_sha256=sha(root/'result.json'),verifier_sha256=sha(Path(__file__)),compiled_files=compiled_files,
    offline_scope='Exact primitive and actualwriter scalar re-computation; completed attention numerical runtime records, source and timing audits; no offline fullattention replay')
(base/'PARTIAL_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k!='compiled_files'},indent=2))
