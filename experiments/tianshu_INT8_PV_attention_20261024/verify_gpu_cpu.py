# SPDX-License-Identifier: Apache-2.0
"""Audit declared configurations, source/codegen, raw medians and all runtime records."""
import argparse,csv,hashlib,json,math,statistics,subprocess
import numpy as np
import domain_cpu
from pathlib import Path
from launch_sweep import CONFIGS
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
base=Path(__file__).resolve().parent;root=a.root.resolve()
assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
assert not (base/'GPU_GATE_32g.json').exists()
data=json.loads((root/'result.json').read_text())
assert data['complete'] and data['phase']=='complete' and data['declared_grid']==CONFIGS
assert data['full_GPU_FP32_and_selected_CPU_FP64_references_passed']
assert all(v is True for v in data['codec_validation'].values())
assert not any(data[k] for k in ('engine_allocator_connected','prefill_executed','model_executed','service_executed','official_benchmark','adopted'))
for path,sha in data['frozen_sources'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((root/'source_snapshot'/Path(path).relative_to('/')).read_bytes()).hexdigest()==sha
assert data['launch_proof']['normalized_launch_AST_unchanged'] and data['launch_proof']['reducer_AST_unchanged']
primitive_counts=dict(quantized_elements=0,integer_outputs=0,FP32_output_words=0)
assert len(data["primitive_records"])==6
assert {(r["T"],r["warps"],r["phase"]) for r in data["primitive_records"]}=={(t,w,p) for t,w in ((32,1),(32,2),(64,2)) for p in range(2)}
for record in data["primitive_records"]:
    path=Path(record["file"]);assert path.is_relative_to(root) and hashlib.sha256(path.read_bytes()).hexdigest()==record["sha256"]
    with np.load(path,allow_pickle=False) as arrays:counts=domain_cpu.verify_primitive(arrays)
    for key,count in counts.items():assert count==record[key];primitive_counts[key]+=count
writer_counts=dict(V_integer_elements=0,V_scale_words=0,K_scale_words=0)
labels=['control']+[c['label'] for c in CONFIGS]
config_by_label={c['label']:c for c in CONFIGS}
assert set(data['compiled_kernel_records'])==set(labels)
compiled_files={}
for label,records in data['compiled_kernel_records'].items():
    assert records
    for r in records:
        assert r['recorded_before_timed_Graph_replay'] and r['target']['backend']=='corex' and r['target']['warp_size']==64
        assert any(path.endswith('.asm') for path in r['files'])
        for path,sha in r['files'].items():
            assert Path(path).is_relative_to(root/'codegen')
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
            compiled_files[path]=sha
assert len(data['cases'])==3
rows=[];by_name={};gpu_cells=cpu_scalars=0
for case in data['cases']:
    assert case['labels']==labels and case['batch']==64
    assert len(case['raw_ms'])==len(labels) and len(case['phases'])==2
    by_name[case['name']]=case
    for index,label in enumerate(labels):
        raw=case['raw_ms'][index]
        assert len(raw)==9 and all(math.isfinite(v) and v>0 for v in raw)
        median=statistics.median(raw)
        assert math.isclose(median,case['median_ms'][index],rel_tol=1e-12)
        cfg=dict(segments=4,TILE_SIZE=16,num_warps=1) if index==0 else config_by_label[label]
        actual=case['actual_launch'][index]
        assert actual['IS_3D'] and actual['sequences']==64 and actual['BLOCK_M']==16 and actual['BLOCK_Q']==2
        assert actual['segments']==cfg['segments'] and actual['tile']==cfg['TILE_SIZE'] and actual['warps']==cfg['num_warps'] and actual['stages']==1
        assert actual['grid'][:2]==case['actual_launch'][0]['grid'][:2] and actual['grid'][2]==cfg['segments']
        assert actual['query_dtype']=='torch.bfloat16' and actual['key_dtype']=='torch.uint8'
        assert actual['value_dtype']==('torch.bfloat16' if index==0 else 'torch.int8')
        assert actual['address_selection']['address_bits']==64
        if index:assert actual["KV_QUANT_MODE"]==2 and actual["PV_INT8"]==cfg["PV_INT8"]
        reduction=100*(1-median/case['median_ms'][0])
        if index:assert math.isclose(reduction,case['candidate_time_reduction_percent'][index-1],abs_tol=1e-10)
        rows.append(dict(case=case['name'],label=label,segments=cfg['segments'],tile=cfg['TILE_SIZE'],warps=cfg['num_warps'],median_ms=median,
            time_reduction_percent=reduction,trial_spread_percent=100*(max(raw)-min(raw))/median,n_regs=actual['n_regs'],n_spills=actual['n_spills']))
    for phase_id,phase in enumerate(case['phases']):
        assert phase['phase']==phase_id and phase['same_Graph_Q_K_V_scales_table_lengths_slots_changed']==bool(phase_id)
        assert len(phase['checks'])==len(labels)
        for index,check in enumerate(phase['checks']):
            assert check['label']==labels[index] and check['quantized_V_reference']==bool(index)
            assert check['physical_page_bytes']==(12288 if index==0 else 8448)
            assert check['numerical']['passed'] and check['numerical']['failed_elements']==0
            assert check['numerical']['atol']==.002 and check['numerical']['rtol']==.01
            for key in ('full_cache_storage_bitwise_exact','output_and_scratch_guards','RNG_unchanged','full_inputs_bitwise_unchanged'):assert check[key]
            if index:
                entry=check["writer_fixture"];path=Path(entry["file"])
                assert path.is_relative_to(root) and hashlib.sha256(path.read_bytes()).hexdigest()==entry["sha256"]
                with np.load(path,allow_pickle=False) as arrays:counts=domain_cpu.verify_writer(arrays)
                for key,count in counts.items():writer_counts[key]+=count
            else:assert check["writer_fixture"] is None
            gpu_cells+=64*16*128
            assert len(check['CPU_FP64'])==12
            for check64 in check['CPU_FP64']:
                assert check64['passed'] and check64['failed_elements']==0 and check64['scalars']==128
                assert check64['atol']==.002 and check64['rtol']==.01
                cpu_scalars+=128
qualified=[]
for index,label in enumerate(labels[1:],1):
    if config_by_label[label]['PV_INT8'] and all(by_name[name]['candidate_time_reduction_percent'][index-1]>=5 for name in ('decode64_4096','decode64_5120')):qualified.append(label)
best={name:min((r for r in rows if r['case']==name and r['label']!='control'),key=lambda r:r['median_ms']) for name in by_name}
gate=dict(complete=True,CPU_only=True,declared_grid_verified=True,all_sources_snapshots_and_compiled_IR_ASM_verified=True,
    all_runtime_numerical_and_changed_data_guards_verified=True,full_GPU_FP32_cells=gpu_cells,selected_CPU_FP64_scalars=cpu_scalars,
    independent_primitive_counts=primitive_counts,independent_writer_counts=writer_counts,
    qualified_fixed_configs_5percent_both_lengths=qualified,best_candidate_by_case=best,
    model_or_service_qualified=False,adopted=False,raw_root=str(root),compiled_files=compiled_files,
    result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    offline_scope='Source/codegen snapshots, runtime reference records and event arithmetic; no full-tensor offline attention recomputation')
with (base/'COMPONENT_KEY_TIMES_32g.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(base/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k!='compiled_files'},indent=2))
