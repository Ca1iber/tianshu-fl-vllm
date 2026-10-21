# SPDX-License-Identifier: Apache-2.0
"""Independent CPU checkpoint/quantization/INT64 selections and all timing recheck."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
import torch
from safetensors import safe_open
BASE=Path('/workspace/vllm-plugin-FL/experiments/tianshu_int8_attention_projections_20261021')
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();out=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not out.exists()
torch.set_num_threads(8);assert not torch.cuda.is_initialized()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
result=json.loads((root/'result.json').read_text());assert not result['complete'] and result['phase']=='failed' and len(result['cases'])==10
assert result['cases'][-1]['name']=='o' and result['cases'][-1]['M']==8
result=dict(result,cases=result['cases'][:9],timing_rounds=[r for r in result['timing_rounds'] if r['case']=='qkv'])
assert not result['adopted'] and not result['model_executed'] and not result['official_benchmark']
for name,digest in result['frozen_sources'].items():
    assert sha(name)==digest==sha(root/'source_snapshot'/Path(name).relative_to('/'))
expected=[(name,m,n,2048) for name,n in [('qkv',2560)] for m in (8,64,128,129,256,512,1024,2048,4096)]
assert [(c['name'],c['M'],c['N'],c['K']) for c in result['cases']]==expected
signatures=json.loads((BASE/'weight_signatures.json').read_text());weights={}
with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as checkpoint:
    for name,record in signatures.items():
        full=torch.cat([checkpoint.get_tensor(key) for key in record['keys']],0).contiguous()
        scale=full.float().abs().amax(1).clamp_min(1e-8)/127.
        packed=torch.round(full.float()/scale[:,None]).clamp(-127,127).to(torch.int8)
        assert tensor_sha(full)==record['BF16_bits_sha256']==result['weights'][name]['BF16_bits_sha256']
        assert tensor_sha(packed)==record['packed_sha256']==result['weights'][name]['packed_sha256']
        assert tensor_sha(scale)==record['scales_sha256']==result['weights'][name]['scales_sha256']
        weights[name]=full
rows=[];checks=0;integer_elements=0;A_elements=0;failures=[];raw_tensor_sha={}
for case in result['cases']:
    m,n,k=case['M'],case['N'],case['K'];valid=[c for c in case['configurations'] if c.get('compiled')]
    assert len(case['configurations'])==20 and valid
    for config in case['configurations']:
        if not config.get('compiled'):failures.append(dict(case=case['name'],M=m,index=config['index'],error=config['compile_error']))
    for phase,path in enumerate(case['phase_files']):
        raw_path=root/path;data=torch.load(raw_path,map_location='cpu',weights_only=True);raw_tensor_sha[str(raw_path)]=sha(raw_path)
        assert data['phase']==phase
        av=data['a'].float();scale=av.abs().amax(1).clamp_min(1e-8)/127.
        q=torch.round(av/scale[:,None]).clamp(-127,127).to(torch.int8)
        assert torch.equal(q,data['actual_A_INT8']) and torch.equal(scale,data['actual_A_scale']);A_elements+=q.numel()
        original=weights[case['name']][data['columns']];assert torch.equal(original,data['original_weight_rows'])
        wf=original.float();ws=wf.abs().amax(1).clamp_min(1e-8)/127.;wq=torch.round(wf/ws[:,None]).clamp(-127,127).to(torch.int8)
        sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.
        assert torch.equal(wq*sign,data['actual_W_INT8']) and torch.equal(ws*factor,data['actual_W_scale'])
        integer=q[data['rows']].to(torch.int64)@(wq*sign).to(torch.int64).T
        assert torch.equal(integer,data['CPU_INT64'])
        reference=integer.float()*scale[data['rows'],None]*(ws*factor)[None,:]
        native_reference=av[data['rows']]@(wf*sign*factor).T
        for config in valid:
            assert len(config['phases'])==3 and config['timing_guards_passed']
            proof=config['phases'][phase]
            assert all(proof[key] for key in ('full_integer_exact','full_quantized_output_passed','CPU_INT64_selected_exact','guards_passed','inputs_unchanged','RNG_unchanged'))
            assert proof['actual_max_sum_abs_integer_products']<2**24
            actual=data['outputs'][str(config['index'])]
            assert torch.equal(actual['integer'].to(torch.int64),integer)
            assert bool(torch.isfinite(actual['candidate']).all()) and not bool(((actual['candidate'].float()-reference).abs()>(.02+.01*reference.abs())).any())
            for key in ('native','linear'):
                assert not bool(((actual[key].float()-native_reference).abs()>(.02+.01*native_reference.abs())).any())
            integer_elements+=integer.numel();checks+=1
    for config in valid:
        timing=[r for r in result['timing_rounds'] if r['case']==case['name'] and r['M']==m and r['index']==config['index']]
        assert len(timing)==7 and sorted(t['round'] for t in timing)==list(range(7))
        for t in timing:
            assert t['replays']==20 and t['order']==[['native','candidate','linear'],['candidate','linear','native'],['linear','native','candidate']][t['round']%3]
            assert all(t[mode+'_ms']>0 for mode in ('native','candidate','linear'))
        native=statistics.median(t['native_ms'] for t in timing);candidate=statistics.median(t['candidate_ms'] for t in timing)
        rows.append(dict(case=case['name'],M=m,N=n,K=k,index=config['index'],configuration=config['configuration'],native_ms=native,candidate_ms=candidate,linear_ms=statistics.median(t['linear_ms'] for t in timing),throughput_gain_percent=100*(native/candidate-1),metadata=config['gemm_metadata']))
best=[max((r for r in rows if r['case']==c['name'] and r['M']==c['M']),key=lambda r:r['throughput_gain_percent']) for c in result['cases']]
assert not torch.cuda.is_initialized()
gate=dict(complete=True,independently_CPU_rechecked=True,CPU_recheck_did_not_initialize_GPU=True,cases=9,original_GPU_run_remains_failed=True,partial_scope='completed_QKV_only',compiled_cases=len(rows),compile_failures=failures,
    CPU_selected_integer_checks=checks,CPU_selected_integer_elements=integer_elements,full_A_quantized_elements=A_elements,full_actual_checkpoint_and_packed_weight_hashes_checked=True,
    timing_groups=len(result['timing_rounds']),dynamic_activation_quantization_included=True,static_weight_packing_excluded=True,
    numerical_reference_is_quantized_weights_and_activations=True,offline_scope_is_selected_outputs_and_full_A_not_full_GPU_outputs=True,
    model_accuracy_not_evaluated=True,service_not_executed=True,adopted=False,best_by_case=best,frozen_sources=result['frozen_sources'],raw_tensor_sha256=raw_tensor_sha,
    raw_root=str(root),raw_result_sha256=sha(root/'result.json'),verifier_sha256=sha(__file__),baseline='Actual QKV/O BF16 mm with original O M2048 tile, auxiliary vendor linear timed')
out.mkdir(parents=True)
(out/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
(out/'GPU_COMPARISON_32g.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'GPU_ROUNDS_32g.json').write_text(json.dumps(result['timing_rounds'],indent=2)+'\n')
with (out/'KEY_TIMINGS_32g.csv').open('w',newline='') as file:
    fields=('case','M','index','native_ms','candidate_ms','linear_ms','throughput_gain_percent','configuration')
    writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
print(json.dumps(dict(complete=True,cases=9,original_GPU_run_remains_failed=True,partial_scope='completed_QKV_only',compiled_cases=len(rows),CPU_checks=checks,full_A_values=A_elements)))
for row in best:print(json.dumps(row))
