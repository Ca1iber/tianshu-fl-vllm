# SPDX-License-Identifier: Apache-2.0
"""Recheck selected integer outputs and all A quantization on CPU from raw archives."""
import argparse,hashlib,json,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    import torch
    torch.set_num_threads(4)
    result=read(root/'result.json');assert result['complete'] and result['phase']=='complete' and len(result['cases'])==12
    assert all(sha(p)==sha(root/'source_snapshot'/Path(p).relative_to('/'))==v for p,v in result['frozen_sources'].items())
    signature={r['prefix']:r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')}
    for name,prefix in [('gateup','model.layers.0.mlp.gate_up_proj'),('lm_head','lm_head')]:
        w=result['weights'][name];ref=signature[prefix]
        assert w['shape']==ref['shape'] and w['BF16_bits_sha256']==ref['original_BF16_sha256'] and w['packed_sha256']==ref['packed_sha256'] and w['scales_sha256']==ref['scales_sha256']
    compiled=0;selected_checks=0;rows=[]
    for case in result['cases']:
        assert case['M'] in (32,37,40,48,56,63) and case['name'] in ('gateup','lm_head')
        for cfg in case['configurations']:
            assert cfg.get('compiled') and len(cfg['phases'])==3 and cfg['timing_guards_passed']
            compiled+=1
            for phase in cfg['phases']:
                assert all(phase[k] for k in ('full_integer_exact','full_quantized_output_passed','CPU_INT64_selected_exact','guards_passed','inputs_unchanged','RNG_unchanged'))
                assert phase['actual_max_sum_abs_integer_products']<2**24
            timing=[r for r in result['timing_rounds'] if r['case']==case['name'] and r['M']==case['M'] and r['index']==cfg['index']]
            assert len(timing)==7 and all(r['replays']==20 and r['native_ms']>0 and r['candidate_ms']>0 for r in timing)
            native=statistics.median(r['native_ms'] for r in timing);candidate=statistics.median(r['candidate_ms'] for r in timing)
            rows.append(dict(role=case['name'],M=case['M'],configuration=cfg['configuration'],index=cfg['index'],native_ms=native,candidate_ms=candidate,speedup_percent=100*(native/candidate-1)))
        for path in case['phase_files']:
            archive=torch.load(root/path,map_location='cpu',weights_only=True)
            a=archive['a'].float();scale=a.abs().amax(1).clamp_min(1.e-8)/127.
            q=torch.round(a/scale[:,None]).clamp(-127,127).to(torch.int8)
            assert torch.equal(q,archive['actual_A_INT8']) and torch.equal(scale,archive['actual_A_scale'])
            exact=q[archive['rows']].to(torch.int64)@archive['actual_W_INT8'].to(torch.int64).T
            assert torch.equal(exact,archive['CPU_INT64'])
            bf16=(exact.float()*scale[archive['rows'],None]*archive['actual_W_scale'][None,:]).to(torch.bfloat16)
            for index,output in archive['outputs'].items():
                assert torch.equal(exact,output['integer'].to(torch.int64))
                assert not bool(((output['candidate'].float()-bf16.float()).abs()>(.02+.01*bf16.float().abs())).any())
                native=a[archive['rows']]@archive['original_weight_rows'].float().T
                # Stored original rows are positive; phases 1/2 negate weights, phase 2 halves them.
                if archive['phase'] in (1,2):native=-native
                if archive['phase']==2:native=native*.5
                assert not bool(((output['native'].float()-native).abs()>(.02+.01*native.abs())).any())
                selected_checks+=1
    assert compiled==48 and selected_checks==144 and len(rows)==48
    best=[]
    for role in ('gateup','lm_head'):
        for m in (32,37,40,48,56,63):best.append(min((r for r in rows if r['role']==role and r['M']==m),key=lambda r:r['candidate_ms']))
    gate=dict(complete=True,independently_CPU_rechecked=True,official_benchmark=False,model_executed=False,adopted=False,
        cases=12,compiled_configurations=compiled,full_GPU_integer_and_quantized_checks=144,CPU_selected_integer_checks=selected_checks,
        all_A_quantization_CPU_rechecked=True,actual_weights_match_qualified_Math=True,all_timed_dynamic_A_quantization_included=True,
        LM_baseline_is_actual_vendor_linear=True,static_W_packing_excluded=True,model_quality_not_evaluated=True,
        frozen_sources=result['frozen_sources'],raw_root=str(root),best_by_role_M=best,
        offline_CPU_scope='All A values and selected W rows/output coordinates; full device matrices were asserted during GPU execution, not archived in full.')
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'ROUNDS_32g.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k not in ('frozen_sources','best_by_role_M')}));print(json.dumps(best))
if __name__=='__main__':main()
