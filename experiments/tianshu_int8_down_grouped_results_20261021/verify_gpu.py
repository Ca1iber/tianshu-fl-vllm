# SPDX-License-Identifier: Apache-2.0
"""Independent CPU recheck of archived selections and alternating event timings."""
import csv,hashlib,json,statistics,sys
from pathlib import Path
import torch
from safetensors import safe_open
BASE=Path(__file__).resolve().parent
root=Path(sys.argv[1]).resolve();result=json.loads((root/'result.json').read_text())
assert result['complete'] and result['phase']=='complete' and len(result['cases'])==9
assert not result['adopted'] and not result['model_executed'] and not result['official_benchmark']
for name,sha in result['frozen_sources'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((root/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==sha
assert [c["M"] for c in result["cases"]]==[8,64,128,129,256,512,1024,2048,4096]
assert all((c["name"],c["N"],c["K"])==("down",2048,6144) for c in result["cases"])
rows=[];cpu_integer_elements=0;cpu_A_elements=0;checks=0;failures=[]
for case in result['cases']:
    m,n,k=case['M'],case['N'],case['K'];valid=[c for c in case['configurations'] if c.get('compiled')]
    assert len(case['configurations'])==12 and valid
    for configuration in case['configurations']:
        if not configuration.get('compiled'):failures.append(dict(case=case['name'],M=m,index=configuration['index'],error=configuration['compile_error']))
    for phase,path in enumerate(case['phase_files']):
        data=torch.load(root/path,map_location='cpu',weights_only=True);assert data['phase']==phase
        a=data['a'].float();scale=a.abs().amax(1).clamp_min(1.e-8)/127.
        q=torch.round(a/scale[:,None]).clamp(-127,127).to(torch.int8)
        assert torch.equal(q,data['actual_A_INT8']) and torch.equal(scale,data['actual_A_scale']);cpu_A_elements+=q.numel()
        with safe_open("/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors",framework="pt",device="cpu") as checkpoint:
            actual_rows=[checkpoint.get_slice("model.layers.0.mlp.down_proj.weight")[int(i):int(i)+1] for i in data["columns"]]
            assert torch.equal(torch.cat(actual_rows,0),data["original_weight_rows"])
        original=data['original_weight_rows'].float();ws=original.abs().amax(1).clamp_min(1.e-8)/127.
        wq=torch.round(original/ws[:,None]).clamp(-127,127).to(torch.int8);sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.
        assert torch.equal(wq*sign,data['actual_W_INT8']) and torch.equal(ws*factor,data['actual_W_scale'])
        integer=q[data['rows']].to(torch.int64)@(wq*sign).to(torch.int64).T
        assert torch.equal(integer,data['CPU_INT64'])
        reference=integer.float()*scale[data['rows'],None]*(ws*factor)[None,:]
        native_ref=a[data['rows']]@(original*sign*factor).T
        for config in valid:
            assert len(config['phases'])==3 and config['timing_guards_passed']
            proof=config['phases'][phase]
            assert all(proof[key] for key in ('full_integer_exact','full_quantized_output_passed','CPU_INT64_selected_exact','guards_passed','inputs_unchanged','RNG_unchanged'))
            assert proof['actual_max_sum_abs_integer_products']<2**24
            actual=data['outputs'][str(config['index'])]
            assert torch.equal(actual['integer'].to(torch.int64),integer)
            assert bool(torch.isfinite(actual['candidate']).all()) and not bool(((actual['candidate'].float()-reference).abs()>(.02+.01*reference.abs())).any())
            assert not bool(((actual['native'].float()-native_ref).abs()>(.02+.01*native_ref.abs())).any())
            cpu_integer_elements+=integer.numel();checks+=1
    for config in valid:
        times=[r for r in result['timing_rounds'] if r['case']==case['name'] and r['M']==m and r['index']==config['index']]
        assert len(times)==7 and sorted(t['round'] for t in times)==list(range(7))
        assert all(t['replays']==20 and t['native_ms']>0 and t['candidate_ms']>0 and t['order']==(['native','candidate'] if t['round']%2==0 else ['candidate','native']) for t in times)
        native=statistics.median(t['native_ms'] for t in times);candidate=statistics.median(t['candidate_ms'] for t in times)
        rows.append(dict(case=case['name'],M=m,N=n,K=k,index=config['index'],configuration=config['configuration'],native_ms=native,candidate_ms=candidate,
            throughput_gain_percent=(native/candidate-1)*100,metadata=config['gemm_metadata']))
best=[]
for case in result['cases']:
    options=[r for r in rows if r['case']==case['name'] and r['M']==case['M']];best.append(max(options,key=lambda x:x['throughput_gain_percent']))
gate=dict(complete=True,independently_CPU_rechecked=True,cases=len(result['cases']),compiled_cases=len(rows),compile_failures=failures,
    CPU_selected_integer_checks=checks,CPU_selected_integer_elements=cpu_integer_elements,full_A_quantized_elements=cpu_A_elements,
    timing_groups=len(result['timing_rounds']),dynamic_activation_quantization_included=True,static_weight_packing_excluded=True,
    numerical_reference_is_quantized_weights_and_activations=True,offline_scope_is_selected_outputs_and_full_A_not_full_GPU_outputs=True,
    model_accuracy_not_evaluated=True,service_not_executed=True,adopted=False,best_by_case=best,
    frozen_sources=result['frozen_sources'],raw_root=str(root),verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'GPU_ROUNDS_32g.json').write_text(json.dumps(result['timing_rounds'],indent=2)+'\n')
(BASE/'GPU_COMPARISON_32g.json').write_text(json.dumps(rows,indent=2)+'\n')
with (BASE/"KEY_TIMINGS_32g.csv").open("w",newline="") as file:
    fields=("M","index","native_ms","candidate_ms","throughput_gain_percent","configuration")
    writer=csv.DictWriter(file,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
gate["baseline"]="Iluvatar BF16 flag_gems.linear, actual Down checkpoint weights"
gate["raw_result_sha256"]=hashlib.sha256((root/"result.json").read_bytes()).hexdigest()
(BASE/"GPU_GATE_32g.json").write_text(json.dumps(gate,indent=2)+"\n")
print(json.dumps(gate))
# Group1 vs group8 use the same M, input and tile; summarize their own medians.
comparisons=[]
for m in (8,64,128,129,256,512,1024,2048,4096):
    selected=[r for r in rows if r['M']==m]
    for i in range(0,12,2):
        a=next((r for r in selected if r['index']==i),None);b=next((r for r in selected if r['index']==i+1),None)
        if a is None or b is None:continue
        assert a['configuration']['group_m']==1 and b['configuration']['group_m']==8
        assert {k:v for k,v in a['configuration'].items() if k!='group_m'}=={k:v for k,v in b['configuration'].items() if k!='group_m'}
        comparisons.append(dict(M=m,tile=i//2,group1_ms=a['candidate_ms'],group8_ms=b['candidate_ms'],group8_speedup_percent=100*(a['candidate_ms']/b['candidate_ms']-1)))
(BASE/'GROUP_COMPARISON_32g.json').write_text(json.dumps(comparisons,indent=2)+'\n')
print(json.dumps(comparisons))
