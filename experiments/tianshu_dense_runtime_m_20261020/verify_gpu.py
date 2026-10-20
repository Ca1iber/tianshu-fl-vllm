# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,json,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    import torch
    torch.set_num_threads(4);result=read(root/'result.json');assert result['complete'] and len(result['cases'])==26
    assert all(sha(p)==sha(root/'source_snapshot'/Path(p).relative_to('/'))==v for p,v in result['frozen_sources'].items())
    groups={};quant={'static':{},'dynamic':{}};summary=[];checks=0
    signature={r['prefix']:r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')}
    for role,prefix in [('gateup','model.layers.0.mlp.gate_up_proj'),('lm_head','lm_head')]:assert all(value==signature[prefix][key] for key,value in result['weights'][role].items())
    for case in result['cases']:
        assert len(case['phases'])==3 and case['timing_guards_passed']
        for phase in case['phases']:
            assert all(phase[k] for k in ('full_INT32_exact','full_BF16_variants_bitwise_equal','full_quantized_output_passed','all_A_quantization_exact','CPU_selected_INT64_exact','guards_passed','RNG_unchanged','inputs_unchanged'))
            assert phase['actual_max_sum_abs_integer_products']<2**24
        key=str((case['role'],sorted(case['configuration'].items())))
        row=groups.setdefault(key,dict(static=set(),dynamic=set(),M=set()))
        row['M'].add(case['M'])
        for name in ('static','dynamic'):
            row[name].add(case['compiled'][name]['gemm']['cache_hash'])
            quant[name].setdefault(case['strided'],set()).add(case['compiled'][name]['quantize']['cache_hash'])
        for path in case['phase_files']:
            data=torch.load(root/path,map_location='cpu',weights_only=True);a=data['a'].float();s=a.abs().amax(1).clamp_min(1.e-8)/127.
            q=torch.round(a/s[:,None]).clamp(-127,127).to(torch.int8);assert torch.equal(q,data['actual_A_INT8']) and torch.equal(s,data['actual_A_scale'])
            w=data['original_W_rows'].float();ws=w.abs().amax(1).clamp_min(1.e-8)/127.;wq=torch.round(w/ws[:,None]).clamp(-127,127).to(torch.int8)
            sign=1 if data['phase']==0 else -1;factor=.5 if data['phase']==2 else 1.
            assert torch.equal(wq*sign,data['actual_W_INT8']) and torch.equal(ws*factor,data['actual_W_scale'])
            exact=q[data['rows']].to(torch.int64)@(wq*sign).to(torch.int64).T;assert torch.equal(exact,data['integer']) and torch.equal(data['static'],data['dynamic'])
            reference=exact.float()*s[data['rows'],None]*(ws*factor)[None,:]
            assert not bool(((data['static'].float()-reference).abs()>(.02+.01*reference.abs())).any());checks+=1
        timings=[r for r in result['timing_rounds'] if r['role']==case['role'] and r['M']==case['M'] and r['strided']==case['strided']]
        assert len(timings)==7 and all(r['replays']==20 for r in timings)
        old=statistics.median(r['static_ms'] for r in timings);new=statistics.median(r['dynamic_ms'] for r in timings)
        summary.append(dict(role=case['role'],M=case['M'],strided=case['strided'],configuration=case['configuration'],static_ms=old,dynamic_ms=new,speedup_percent=100*(old/new-1)))
    assert checks==78
    for row in groups.values():assert len(row['dynamic'])==1 and len(row['static'])==len(row['M'])
    assert all(len(values)==1 for values in quant['dynamic'].values())
    compact=[dict(group=key,static_binaries=len(row['static']),dynamic_binaries=len(row['dynamic']),M=sorted(row['M'])) for key,row in groups.items()]
    gate=dict(complete=True,independently_CPU_rechecked=True,model_executed=False,adopted=False,official_benchmark=False,
        full_device_variant_INT32_and_BF16_bitwise_checks=78,CPU_selected_INT64_checks=checks,all_A_and_selected_W_quantization_recomputed=True,
        one_GEMM_binary_per_role_configuration_across_all_M_proven=True,one_quantize_binary_per_stride_across_all_M_proven=True,
        binary_reuse=compact,quantize_binary_counts={name:{str(stride):len(values) for stride,values in rows.items()} for name,rows in quant.items()},
        frozen_sources=result['frozen_sources'],raw_root=str(root),summary=summary,
        scope='Same tile and integer arithmetic only; no new service/Math or formal throughput. Offline CPU checks selected outputs; full device matrices compared during GPU execution.')
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'ROUNDS_32g.json').write_text(json.dumps(result['timing_rounds'],indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k not in ('frozen_sources','summary')}));print(json.dumps(summary))
if __name__=='__main__':main()
