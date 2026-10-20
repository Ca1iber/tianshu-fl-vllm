# SPDX-License-Identifier: Apache-2.0
"""Independent CPU archive checker, never replays GPU work."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True);args=parser.parse_args();root=args.raw
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    result=json.loads((root/'result.json').read_text());assert result['phase']=='GPU_complete' and result['component_only'] and not result['adopted']
    cpu=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert sha(BASE/'CPU_GATE_32g.json')==result['CPU_gate_sha256'] and cpu['source_sha256']==result['source_sha256']
    for path,digest in result['source_sha256'].items():assert sha(path)==digest
    assert result['compiled_trace_sources']
    for path,digest in result['compiled_trace_sources'].items():assert sha(path)==digest
    for name,meta in result['fixtures'].items():assert sha(root/name)==meta['sha256'] and (root/name).stat().st_size==meta['bytes']
    with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as weights:weight=weights.get_tensor('model.layers.0.post_attention_layernorm.weight')
    assert hashlib.sha256(weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==result['weight_sha256']
    comparisons=[];full_quantized_elements=0;selected_math_elements=0;rows=[]
    def check_quant(data,m):
        nonlocal full_quantized_elements
        assert data['norm'].shape==data['q'].shape==(m,2048) and data['scales'].shape==(m,)
        value=data['norm'].float();s=value.abs().amax(1).clamp_min(1.e-8)/127.;q=torch.round(value/s[:,None]).clamp(-127,127).to(torch.int8)
        assert torch.equal(s,data['scales']) and torch.equal(q,data['q']);full_quantized_elements+=q.numel()
    assert [c['M'] for c in result['cases']]==[8,64,4096]
    for case in result['cases']:
        m=case['M'];assert len(case['phases'])==3 and len(case['configurations'])==3 and len(case['rounds'])==7
        baseline=torch.load(root/case['baseline_fixture'],map_location='cpu',weights_only=True);assert torch.equal(baseline['weight'],weight)
        check_quant(baseline,m)
        selected=torch.tensor(sorted({0,1,2,m//2,m-1}));combined=baseline['x'][selected].double()+baseline['residual'][selected].double()
        exact=combined*torch.rsqrt((combined*combined).mean(-1,keepdim=True)+1.e-6)*weight.double()
        eager=((combined.float()*torch.rsqrt((combined.float()*combined.float()).mean(-1,keepdim=True)+1.e-6)).to(torch.bfloat16)*weight).to(torch.bfloat16)
        # Current compiled code is allowed to elide the IR's intermediate cast.
        comparisons.append(dict(M=m,native_IR_vs_compiled_selected_unequal=int((eager!=baseline['norm'][selected]).sum())))
        for phase in case['phases']:
            assert len(phase['variants'])==3
            for item in phase['variants']:
                assert item['full_quantization_and_scales_exact_from_materialized_BF16'] and item['compiled_reference_tolerance_passed'] and item['guards_preserved']
                assert item['maximum_quant_difference']<=2
        for index,config in enumerate(case['configurations']):
            item=case['phases'][2]['variants'][index];data=torch.load(root/item['fixture'],map_location='cpu',weights_only=True);check_quant(data,m)
            actual=data['norm'][selected].double();bound=1.e-5+.015625*exact.abs();assert bool(torch.isfinite(actual).all()) and not bool(((actual-exact).abs()>bound).any());selected_math_elements+=actual.numel()
            difference=(data['norm'].float()-baseline['norm'].float()).abs();limit=1.e-5+.015625*baseline['norm'].float().abs();assert not bool((difference>limit).any())
            assert int((data['q'].short()-baseline['q'].short()).abs().max())<=2
            assert int((data['norm']!=baseline['norm']).sum())==item['compiled_norm_unequal_elements'] and int((data['q']!=baseline['q']).sum())==item['compiled_quant_unequal_elements']
            name=f'fused_{index}'
            for event in case['rounds']:assert event['replays']==20 and set(event['order'])=={'baseline','fused_0','fused_1','fused_2'} and all(v>0 for v in event['milliseconds'].values())
            native=statistics.median(e['milliseconds']['baseline'] for e in case['rounds']);candidate=statistics.median(e['milliseconds'][name] for e in case['rounds'])
            assert native==case['median_ms']['baseline'] and candidate==case['median_ms'][name]
            rows.append(dict(M=m,index=index,configuration=config['configuration'],baseline_ms=native,candidate_ms=candidate,speed_gain_percent=100*(native/candidate-1),n_regs=config['n_regs'],n_spills=config['n_spills'],compiled_norm_unequal_elements=item['compiled_norm_unequal_elements'],compiled_quant_unequal_elements=item['compiled_quant_unequal_elements']))
    assert result['inputs_and_CPU_GPU_RNG_preserved'] and result['full_three_phase_checks']==27 and not torch.cuda.is_initialized()
    gate=dict(complete=True,CPU_only_independent_recheck=True,component_only=True,adopted=False,model_not_installed=True,Math_not_run=True,
        CPU_full_phase2_quantized_elements=full_quantized_elements,CPU_selected_FP64_RMS_elements=selected_math_elements,
        GPU_three_phase_checks=27,compiled_baseline_is_not_bitwise_equivalent_claim=True,new_model_quality_required=True,
        rows=rows,native_IR_vs_compiled_selected=comparisons,raw_root=str(root),raw_result_sha256=sha(root/'result.json'),verifier_sha256=sha(__file__),source_sha256=result['source_sha256'],compiled_trace_sources=result['compiled_trace_sources'])
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'ROUNDS_32g.json').write_text(json.dumps([dict(M=c['M'],rounds=c['rounds']) for c in result['cases']],indent=2)+'\n')
    with (BASE/'KEY_TIMINGS_32g.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(gate))
if __name__=='__main__':main()
