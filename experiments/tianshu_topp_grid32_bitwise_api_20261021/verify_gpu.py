# SPDX-License-Identifier: Apache-2.0
"""Reconstruct every FP32 GPU output and independently check full FP64 row semantics."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
import torch
from artifacts import tensor_sha
from semantic_gate import verdict
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();micro=root/'micro';out=a.output.resolve()
assert root.is_relative_to('/workspace/benchmark-results') and not out.exists()
torch.set_num_threads(8);assert not torch.cuda.is_initialized()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def same_number(x,y):return x==y or (math.isfinite(x) and math.isfinite(y) and abs(x-y)<=2e-12)
data=read(micro/'result.json');gate=read(Path(__file__).with_name('CPU_GATE_32g.json'))
assert data['complete'] and data['phase']=='complete'
assert (root/'complete.txt').is_file() and (root/'exit_code.txt').read_text().strip()=='0'
assert len(data['masks'])==300 and len(data['RNG_cases'])==33 and len(data['events'])==500
assert len(data['fallbacks'])==8 and len(data['lifecycle_cases'])==8
assert len(data['profile_environment'])==62 and len(data['profile_frozen_sources'])==gate['runtime_source_count']
assert data['profile_frozen_sources']==gate['profiles'][1]['sources']
for path,h in data['source_snapshots'].items():
    assert sha(path)==h==sha(micro/'source_snapshot'/Path(path).relative_to('/'))
inputs={};file_hashes={};checks=0;full_values=0;strict_bad=[];changed=0;timings=[];layout_cases=set()
for case,record in data['input_artifacts'].items():
    path=micro/record['path'];assert sha(path)==record['sha256'];file_hashes[str(path)]=record['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert tensor_sha(saved['values'],torch)==record['tensor_sha256']==saved['sha256']
    assert list(saved['values'].shape)==record['shape'] and saved['original_stride']==record['original_stride']
    inputs[int(case)]=saved['values']
assert len(inputs)==25
for row in data['masks']:
    case,pindex,phase=row['case'],row['probability_index'],row['phase']
    assert pindex in range(6) and phase in (0,1)
    artifact=row['full_output_artifact'];path=micro/artifact['path']
    assert sha(path)==artifact['sha256'];file_hashes[str(path)]=artifact['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert (saved['case'],saved['pindex'],saved['phase'])==(case,pindex,phase)
    values=inputs[case].roll(17*(pindex+phase),dims=1)
    assert list(values.shape)==[row['batch'],130560]
    expected_p=torch.full((row['batch'],),(.0001,.1,.5,.95,.999,1.)[pindex],dtype=torch.float32)
    assert torch.equal(saved['p'],expected_p)
    probabilities=values.double().softmax(-1);p_cpu=saved['p'].double()
    masks={}
    for name,record in saved['outputs'].items():
        assert record['shape']==list(values.shape) and record['packed_finite'].dtype==torch.uint8
        bits=np.unpackbits(record['packed_finite'].numpy(),bitorder='little')
        assert len(bits)==values.numel()
        finite=torch.from_numpy(bits.reshape(values.shape).astype(bool));masks[name]=finite
        output=torch.where(finite,values,torch.tensor(-math.inf,dtype=torch.float32))
        assert tensor_sha(output,torch)==record['full_FP32_sha256']
        actual_stride=[130567,1] if row['distribution'].endswith('strided') else [130560,1]
        assert record['actual_stride']==actual_stride
        if row['distribution'].endswith('strided'):layout_cases.add((case,phase,name))
        full_values+=values.numel()
    assert set(masks)=={'original','candidate'}
    assert torch.equal(masks['original'],masks['candidate']), ('all_full_FP32_outputs_bitwise_original',case,pindex,phase)
    for name,finite in masks.items():
        mass=(probabilities*finite).sum(1)
        smallest=torch.where(finite,probabilities,math.inf).min(1).values
        low=torch.where(finite,values.double(),math.inf).min(1).values
        high=torch.where(~finite,values.double(),-math.inf).max(1).values
        nonempty=finite.any(1);exact=(finite==masks['original']).all(1)
        summary=row['checks'][name]
        for index in range(row['batch']):
            actual=verdict(float(mass[index]),float(smallest[index]),float(p_cpu[index]),float(low[index]),float(high[index]),bool(nonempty[index]),bool(exact[index]),True)
            recorded=summary['row_checks'][index]
            assert all(actual[k]==recorded[k] for k in actual)
            assert actual['relative_gate']
            assert same_number(float(mass[index]),summary['kept_mass'][index])
            assert same_number(float(smallest[index]),summary['smallest_kept_probability'][index])
            assert float(p_cpu[index])==summary['CPU_p'][index]
            if not actual['strict_CPU']:strict_bad.append(dict(case=case,pindex=pindex,phase=phase,implementation=name,row=index,exact_current_baseline=actual['exact_current_baseline']))
            if name=='candidate' and not actual['exact_current_baseline']:changed+=1
            checks+=1
    if row['probability']==.95:
        for name in ('original','candidate'):
            events=[r['ms'] for r in data['events'] if r['case']==case and r['phase']==phase and r['mode']==name]
            assert len(events)==5 and statistics.median(events)==row['medians_ms'][name]
        timings.append(dict(case=case,batch=row['batch'],distribution=row['distribution'],phase=phase,original_ms=row['medians_ms']['original'],candidate_ms=row['medians_ms']['candidate'],time_reduction_percent=100*(1-row['medians_ms']['candidate']/row['medians_ms']['original'])))
expected_batches={8,16,24,31,32,33,40,48,56,63,64}
assert {(r['batch'],r['seeded']) for r in data['RNG_cases']}=={(b,s) for b in expected_batches for s in ('none','partial','all')}
for r in data['RNG_cases']:
    assert r['global_and_request_states_identical'] and r['sampled_from_retained_logits']
    assert r['global_after_sha256']['original']==r['global_after_sha256']['candidate']
    assert r['request_after_sha256']['original']==r['request_after_sha256']['candidate']
    for tokens in r['sampled_tokens'].values():assert len(tokens)==r['batch'] and all(0<=t<130560 for t in tokens)
    if r['identical_mask']:assert r['identical_tokens']
    assert r['identical_mask'] and r['identical_tokens']
assert [r['batch'] for r in data['lifecycle_cases']]==[64,8,32,33,64,8,32,33]
assert all(all(r[k] for k in ('input_changed','same_graph_changed_input_exact','older_other_batch_outputs_preserved','RNG_unchanged')) for r in data['lifecycle_cases'])
assert all(r['bitwise'] for r in data['fallbacks'])
assert [(r['batch'],r['topk'],r['topp']) for r in data['fallbacks']]==[(1,False,True),(2,False,True),(4,False,True),(7,False,True),(65,False,True),(64,False,False),(64,True,False),(64,True,True)]
records=list((micro/'grid_topp_records').glob('actual_*.json'));assert len(records)==1
routes=read(records[0])['routes'];active=[r for r in routes.values() if r['active']]
assert {r['batch'] for r in active}==expected_batches
assert {(r['batch'],r['input_row_stride']) for r in active if r['input_row_stride']==130567}=={(8,130567),(64,130567)}
for r in active:
    m=r['actual_candidate'];batch=r['batch'];original_grid=min(batch,16)
    assert m['source_sha256']=='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
    assert m['BLOCK_SIZE']==32768 and m['BLOCK_SIZE_TRUNC']==16384 and m['num_warps']==16 and m['num_stages']==1
    assert m['grid']==[min(batch,original_grid*2)] and m['original_grid']==[original_grid]
    assert len(m['binary_sha256'])==64 and m['actual_buffer_rows']>=(m['grid'][0])
artifact_dir=micro/'grid_topp_records/kernel_artifacts/fixed32_grid2'
manifest=read(artifact_dir/'manifest.json')
assert manifest['metadata']['source_sha256']=='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
for name,digest in manifest['files'].items():assert sha(artifact_dir/name)==digest
assert len(layout_cases)==8 and not torch.cuda.is_initialized()
assert changed==0
report=dict(complete=True,independently_CPU_recomputed=True,CPU_recompute_did_not_initialize_GPU=True,adopted=False,official_benchmark=False,model_service_not_run=True,
 parent_runtime_sources=407,runtime_source_count=gate['runtime_source_count'],environment_count=62,API_mask_cases=300,input_distributions=25,input_phases=2,
 full_FP32_outputs_reconstructed=600,full_FP32_values=full_values,independent_FP64_row_checks=checks,all_per_row_original_contract_passed=True,
 strict_CPU_false_rows_preserved=len(strict_bad),strict_false_rows_all_equal_baseline=all(r['exact_current_baseline'] for r in strict_bad),strict_false_rows=strict_bad,
 all_600_full_FP32_outputs_bitwise_original=True,all_33_actual_sampler_masks_tokens_and_RNG_bitwise_original=True,candidate_changed_rows=changed,RNG_cases=33,RNG_validation_scope='Instrumented actual sampler assertions and recorded state hashes; not re-executed by CPU verifier',
 same_graph_lifecycle_cases=8,original_fallbacks=8,actual_strided_output_cases=len(layout_cases),timing_events=500,timings=timings,
 actual_active_routes=active,compiled_contiguous_B64_artifact_manifest=manifest,raw_tensor_sha256=file_hashes,
 raw_result_path=str(micro/'result.json'),raw_result_sha256=sha(micro/'result.json'),verifier_sha256=sha(__file__),frozen_sources=data['source_snapshots'])
out.mkdir(parents=True)
(out/'GPU_GATE_32g.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
with (out/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(timings[0]),lineterminator='\n');writer.writeheader();writer.writerows(timings)
print(json.dumps(dict(complete=True,outputs=600,full_values=full_values,FP64_row_checks=checks,strict_false_preserved=len(strict_bad),changed_rows=changed)))
for r in timings:
    if r['batch'] in (8,64) and r['distribution'] in ('bf16_normal','bf16_peaked','bf16_strided'):print(json.dumps(r))
