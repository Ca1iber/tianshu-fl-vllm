# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,json,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    import torch
    torch.set_num_threads(4);result=read(root/'result.json');assert result['complete'] and result['phase']=='complete' and len(result['cases'])==26
    assert all(sha(p)==sha(root/'source_snapshot'/Path(p).relative_to('/'))==v for p,v in result['frozen_sources'].items())
    expected={r['prefix']:r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')}
    for role,prefix in [('gateup','model.layers.0.mlp.gate_up_proj'),('lm_head','lm_head')]:assert all(value==expected[prefix][key] for key,value in result['weights'][role].items())
    checks=0;summary=[];groups={};quant={}
    for case in result['cases']:
        assert case['padded_M']==(32 if case['M']<=32 else 64) and len(case['phases'])==3 and case['timing_guards_passed']
        for phase in case['phases']:
            assert all(phase[k] for k in ('full_INT32_exact','full_BF16_variants_bitwise_equal','full_quantized_output_passed','all_A_quantization_exact','CPU_selected_INT64_exact','guards_passed','RNG_unchanged','inputs_unchanged','padded_rows_zero'))
        key=str((case['role'],case['padded_M'],sorted(case['configuration'].items())))
        group=groups.setdefault(key,dict(cache=set(),binary=set(),M=set()))
        group['cache'].add(case['compiled']['dynamic']['gemm']['cache_hash']);group['binary'].add(case['compiled']['dynamic']['gemm']['binary_sha256']);group['M'].add(case['M'])
        if case['M']!=case['padded_M']:
            row=quant.setdefault(case['strided'],dict(cache=set(),binary=set(),M=set()))
            row['cache'].add(case['compiled']['dynamic']['quantize']['cache_hash']);row['binary'].add(case['compiled']['dynamic']['quantize']['binary_sha256']);row['M'].add(case['M'])
        for filename in case['phase_files']:
            data=torch.load(root/filename,map_location='cpu',weights_only=True)
            value=data['a'].float();s=value.abs().amax(1).clamp_min(1.e-8)/127.;q=torch.round(value/s[:,None]).clamp(-127,127).to(torch.int8)
            assert torch.equal(q,data['actual_A_INT8']) and torch.equal(s,data['actual_A_scale'])
            value=data['original_W_rows'].float();ws=value.abs().amax(1).clamp_min(1.e-8)/127.;wq=torch.round(value/ws[:,None]).clamp(-127,127).to(torch.int8)
            sign=1 if data['phase']==0 else -1;factor=.5 if data['phase']==2 else 1.
            assert torch.equal(wq*sign,data['actual_W_INT8']) and torch.equal(ws*factor,data['actual_W_scale'])
            exact=q[data['rows']].to(torch.int64)@(wq*sign).to(torch.int64).T;assert torch.equal(exact,data['integer']) and torch.equal(data['static'].view(torch.int16),data['dynamic'].view(torch.int16))
            reference=exact.float()*s[data['rows'],None]*(ws*factor)[None,:]
            assert not bool(((data['static'].float()-reference).abs()>(.02+.01*reference.abs())).any());checks+=1
        timings=[r for r in result['timing_rounds'] if r['role']==case['role'] and r['M']==case['M'] and r['strided']==case['strided']]
        assert len(timings)==7 and all(r['replays']==20 and r['static_ms']>0 and r['dynamic_ms']>0 for r in timings)
        direct=statistics.median(r['static_ms'] for r in timings);padded=statistics.median(r['dynamic_ms'] for r in timings)
        summary.append(dict(role=case['role'],M=case['M'],padded_M=case['padded_M'],strided=case['strided'],same_configuration=case['configuration'],direct_ms=direct,padded_ms=padded,speedup_percent=100*(direct/padded-1)))
    assert checks==78 and all(len(row['cache'])==len(row['binary'])==1 for row in groups.values()) and all(len(row['cache'])==len(row['binary'])==1 for row in quant.values())
    gate=dict(complete=True,independently_CPU_rechecked=True,official_benchmark=False,model_executed=False,adopted=False,
        full_device_INT32_and_BF16_valid_row_bitwise_checks=78,CPU_selected_INT64_checks=checks,all_A_and_selected_W_quantization_recomputed=True,
        padded_rows_written_zero_inside_timed_quantizer=True,one_padded_GEMM_binary_per_role_target=True,one_padded_quantize_binary_per_stride=True,
        actual_equals_target_keeps_original_quantizer=True,alignment_contract_bytes=256,
        padding_reuse=[dict(group=key,M=sorted(row['M']),compiled_GEMM_binaries=len(row['binary'])) for key,row in groups.items()],
        quantize_reuse=[dict(strided=key,M=sorted(row['M']),compiled_binaries=len(row['binary'])) for key,row in quant.items()],
        frozen_sources=result['frozen_sources'],raw_root=str(root),summary=summary,
        scope='Same target tile, direct actual-M INT8 vs padded INT8; all A quantization/padded writes included. No BF16/service/Math throughput assertion. CPU checks selected outputs, device checks full valid matrices.')
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'ROUNDS_32g.json').write_text(json.dumps(result['timing_rounds'],indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k not in ('frozen_sources','summary')}));print(json.dumps(summary))
if __name__=='__main__':main()
