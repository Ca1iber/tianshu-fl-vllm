# SPDX-License-Identifier: Apache-2.0
"""CPU independently requantizes all A/selected original W and INT64 selected dots."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4);result=read(root/'result.json');assert result['complete'] and result['phase']=='complete' and len(result['cases'])==28
    assert all(sha(p)==sha(root/'source_snapshot'/Path(p).relative_to('/'))==v for p,v in result['frozen_sources'].items())
    expected={r['prefix']:r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')}
    for role,prefix in [('gateup','model.layers.0.mlp.gate_up_proj'),('lm_head','lm_head')]:assert all(result['weights'][role][k]==expected[prefix][k] for k in ('original_BF16_sha256','packed_sha256','scales_sha256','shape'))
    plan={(r['case'],r['M']):r['configuration'] for r in read(BASE.parent/'tianshu_dense_w8a8_20261020/MODEL_INTEGRATION_PLAN_32g.json')['selected_configurations']}
    checks=0;summary=[];artifacts=[]
    with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as checkpoint:
      for case in result['cases']:
        assert case['allocated_M']==case['M'] and len(case['phases'])==3 and case['timing_guards_passed']
        assert case['configuration']==plan[(case['role'],8 if case['M']<=32 else 64)]
        if case['ragged']:assert (case['M'],case['N'],case['K'],case['strided'])==(17,130,259,True)
        for phase in case['phases']:
            assert all(phase[k] for k in ('full_INT32_exact','full_BF16_variants_bitwise_equal','full_W_NN_transpose_bytes_verified','full_quantized_output_passed','all_A_quantization_exact','CPU_selected_INT64_exact','guards_passed','RNG_unchanged','inputs_unchanged'))
            assert phase['actual_max_sum_abs_integer_products']<2**24
        for index,filename in enumerate(case['phase_files']):
            path=root/filename;data=torch.load(path,map_location='cpu',weights_only=True);assert data['phase']==index and data['K']==case['K']
            if case['role']=='gateup':
                def row_for(i):return checkpoint.get_slice('model.layers.0.mlp.'+('gate_proj' if i<6144 else 'up_proj')+'.weight')[i%6144:i%6144+1]
                original=torch.cat([row_for(int(i)) for i in data['columns']],0)
            else:original=torch.cat([checkpoint.get_slice('lm_head.weight')[int(i):int(i)+1] for i in data['columns']],0)
            assert torch.equal(original,data['original_W_rows'])
            value=data['a'].float();s=value.abs().amax(1).clamp_min(1.e-8)/127.;q=torch.round(value/s[:,None]).clamp(-127,127).to(torch.int8)
            assert torch.equal(q,data['actual_A_INT8']) and torch.equal(s,data['actual_A_scale'])
            value=original.float();ws=value.abs().amax(1).clamp_min(1.e-8)/127.;wq=torch.round(value/ws[:,None]).clamp(-127,127).to(torch.int8)[:,:case['K']]
            sign=1 if data['phase']==0 else -1;factor=.5 if data['phase']==2 else 1.
            assert torch.equal(wq*sign,data['actual_W_INT8']) and torch.equal(ws*factor,data['actual_W_scale'])
            exact=q[data['rows']].to(torch.int64)@(wq*sign).to(torch.int64).T;assert torch.equal(exact,data['integer']) and torch.equal(data['static'].view(torch.int16),data['dynamic'].view(torch.int16))
            reference=exact.float()*s[data['rows'],None]*(ws*factor)[None,:]
            assert not bool(((data['static'].float()-reference).abs()>(.02+.01*reference.abs())).any());checks+=1
            artifacts.append(dict(file=filename,bytes=path.stat().st_size,sha256=sha(path)))
        timings=[r for r in result['timing_rounds'] if all(r[k]==case[k] for k in ('role','M','strided','ragged'))]
        assert len(timings)==7 and [r['round'] for r in timings]==list(range(7))
        assert all(r['replays']==20 and r['order']==(['static','dynamic'] if r['round']%2==0 else ['dynamic','static']) and all(math.isfinite(r[k]) and r[k]>0 for k in ('static_ms','dynamic_ms')) for r in timings)
        nt=statistics.median(r['static_ms'] for r in timings);nn=statistics.median(r['dynamic_ms'] for r in timings)
        summary.append(dict(role=case['role'],M=case['M'],N=case['N'],K=case['K'],strided=case['strided'],ragged=case['ragged'],same_configuration=case['configuration'],NT_ms=nt,NN_ms=nn,speedup_percent=100*(nt/nn-1),NT_metadata=case['compiled']['static'],NN_metadata=case['compiled']['dynamic']))
    assert checks==84 and len(result['timing_rounds'])==196
    gate=dict(complete=True,independently_CPU_rechecked=True,official_benchmark=False,model_executed=False,adopted=False,
        full_device_INT32_and_BF16_bitwise_checks=84,CPU_selected_INT64_checks=checks,all_A_and_selected_original_checkpoint_W_quantization_recomputed=True,
        full_W_NN_transpose_bytes_verified_on_device=True,alignment_contract_bytes=256,N130_K259_dual_tail_and_strided_cases_included=True,
        timing_groups=196,replays_per_group=20,static_weight_transpose_and_H2D_excluded_from_timing=True,
        frozen_sources=result['frozen_sources'],raw_root=str(root),raw_result_sha256=sha(root/'result.json'),summary=summary,artifacts=artifacts,
        scope='Same selected tile and actual-M A quantization, NT integer vs NN integer; not BF16 baseline, model, Math or official score. CPU selected outputs, device full matrices.')
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'ROUNDS_32g.json').write_text(json.dumps(result['timing_rounds'],indent=2)+'\n')
    with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as file:
        keys=('role','M','N','K','strided','ragged','NT_ms','NN_ms','speedup_percent');writer=csv.DictWriter(file,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in summary)
    print(json.dumps({k:v for k,v in gate.items() if k not in ('frozen_sources','summary','artifacts')}));print(json.dumps(summary))
if __name__=='__main__':main()
