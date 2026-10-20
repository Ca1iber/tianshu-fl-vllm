# SPDX-License-Identifier: Apache-2.0
"""Offline: source/axis provenance, full output tensors, CPU GQA and timing."""
import csv,hashlib,json,math,statistics,sys
from pathlib import Path
import torch
import source
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_k8v16_address32_20261020/qualified_S16_reader.py'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def validate(actual,expected):
    a,e=actual.float(),expected.float();assert torch.isfinite(a).all() and torch.isfinite(e).all()
    delta=(a-e).abs();assert not (delta>.002+.01*e.abs()).any()
    return float(delta.max())

def main():
    root=Path(sys.argv[1]);data=read(root/'result.json');torch.set_num_threads(1)
    assert data['complete'] and data['phase']=='complete' and data['numerical_passed']
    assert data['same_Graph_Q_K_V_table_lengths_verified'] and not data['adoption_allowed']
    assert not data['service_executed'] and not data['model_executed'] and not data['official_benchmark']
    for path,digest in data['frozen_sources'].items():
        assert sha(path)==sha(root/'source_snapshot'/Path(path).relative_to('/'))==digest
    assert 'No running processes found' in (root/'hardware_before.txt').read_text()
    labels=list(source.ORDERS);assert set(data['model_sources'])==set(labels)
    for label,proof in data['model_sources'].items():
        generated,expected=source.render(PARENT,label)
        assert Path(proof['generated_source']).read_text()==generated
        assert all(proof[k]==v for k,v in expected.items())
    assert len(data['cases'])==12 and len({r['name'] for r in data['cases']})==12
    rows=[];artifacts=[];gpu=cpu=recorded_cpu=elements=0;maxerr=0.
    for case in data['cases']:
        assert case['labels']==labels and len(case['phases'])==2
        n=case['batch'];previous=None
        for phase,row in enumerate(case['phases']):
            assert row['phase']==phase and row['Q_K_V_table_lengths_slots_changed_in_place']==bool(phase)
            info=row['output_artifact'];path=Path(info['path'])
            assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=True)
            assert saved['q'].shape==(n,16,128) and saved['expected_fp32'].shape==(n,16,128)
            assert set(saved['outputs'])==set(labels) and saved['lengths'].tolist()==row['lengths']
            if previous is not None:
                assert not torch.equal(saved['q'],previous['q']) and not torch.equal(saved['lengths'],previous['lengths'])
                assert not torch.equal(saved['table'],previous['table'])
            previous=saved
            reference=saved['expected_fp32']
            for sequence,payload in saved['selected_cpu_KV'].items():
                length=int(saved['lengths'][sequence]);key,value=payload['k'].float(),payload['v'].float()
                assert key.shape==value.shape==(length,2,128)
                for head in (0,7,8,15):
                    scores=(key[:,head//8]@saved['q'][sequence,head].float())*(128**-.5)
                    answer=scores.softmax(0)@value[:,head//8]
                    maxerr=max(maxerr,validate(reference[sequence,head],answer));cpu+=1
            assert len(row['checks'])==len(labels)
            for label,check in zip(labels,row['checks']):
                assert check['label']==label and check['RNG_unchanged'] and check['all_storage_bitwise_exact'] and check['output_and_segment_guards']
                assert check['physical_page_bytes']==12288 and check['V_max_quantization_abs']==0
                assert check['numerical']['passed'] and check['numerical']['failed_elements']==0
                assert all(c['passed'] and c['failed_elements']==0 for c in check['cpu_checks'])
                observed=saved['outputs'][label];assert observed.dtype==torch.bfloat16 and observed.shape==reference.shape
                delta=validate(observed,reference)
                assert abs(delta-check['numerical']['max_abs'])<1e-9
                assert torch.equal(observed,saved['outputs']['current'])
                elements+=observed.numel();gpu+=1;recorded_cpu+=len(check['cpu_checks']);maxerr=max(maxerr,delta)
            artifacts.append(dict(case=case['name'],phase=phase,**info))
        assert len(case['raw_ms'])==4 and all(len(x)==9 and all(math.isfinite(v) and v>0 for v in x) for x in case['raw_ms'])
        medians=[statistics.median(values) for values in case['raw_ms']];assert medians==case['median_ms']
        assert case['candidate_speedup_percent']==[100*(medians[0]/v-1) for v in medians[1:]]
        before=(n//2+n,2,case['actual_launch'][0]['segments'])
        for index,(label,record) in enumerate(zip(labels,case['actual_launch'])):
            assert tuple(record['grid'])==before and record['IS_3D'] and record['key_dtype']=='torch.uint8' and record['value_dtype']=='torch.bfloat16'
            order=record['cta_order'];assert order['mode']==label and order['IS_3D']
            assert order['grid_before']==list(before) and order['grid_after']==list(source.remap_grid(before,label))
            assert order['num_warps']==record['warps'] and order['num_stages']==record['stages']==1
            assert record['tile']==case['actual_launch'][0]['tile'] and record['warps']==case['actual_launch'][0]['warps']
            rows.append(dict(case=case['name'],variant=label,batch=n,strided=case['strided'],median_ms=medians[index],
                speedup_percent=0 if index==0 else case['candidate_speedup_percent'][index-1],
                original_grid=str(list(before)),actual_grid=str(order['grid_after'])))
    assert gpu==96 and cpu==256 and recorded_cpu==1024
    gate=dict(complete=True,independently_rechecked=True,component_only=True,adopted=False,
        full_GPU_FP32_output_checks=gpu,independently_recomputed_CPU_reference_vectors=cpu,
        CPU_reference_checks_recorded=recorded_cpu,full_output_elements_rechecked=elements,max_abs_vs_references=maxerr,
        all_candidate_output_tensors_bitwise_equal_to_current=True,actual_permuted_grid_and_qualified_launch_verified=True,
        full_cache_storage_and_scratch_guards_asserted_in_frozen_probe=True,Graph_replay_RNG_unchanged=True,
        changed_data_same_Graph_phases_verified=True,timing_groups=432,replays_per_group=20,
        model_or_quality_or_service_not_run=True,raw_root=str(root),raw_result_sha256=sha(root/'result.json'),
        frozen_sources=data['frozen_sources'],model_sources=data['model_sources'],cases=rows,
        payloads_retained_in_container_only=artifacts,verifier_source_sha256=sha(__file__))
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    (root/'independent_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    with (BASE/'GPU_ROUNDS_32g.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in gate.items() if k not in ('frozen_sources','model_sources','cases','payloads_retained_in_container_only')}))
if __name__=='__main__':main()
