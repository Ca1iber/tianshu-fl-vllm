# SPDX-License-Identifier: Apache-2.0
"""Independent CPU recheck of actual GPU selected integers, weights and timing."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent
PROBE=BASE.parent/'tianshu_int8_nn_gateup_tiles_20261021'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4);result=read(root/'result.json');assert result['complete'] and result['phase']=='complete'
    assert all(sha(p)==sha(root/'source_snapshot'/Path(p).relative_to('/'))==v for p,v in result['frozen_sources'].items())
    tiles=read(PROBE/'tiles.json');assert len(tiles)==17 and len(result['cases'])==34
    assert sorted((r['M'],r['tile_id']) for r in result['cases'])==[(m,i) for m in (8,64) for i in range(17)]
    expected=next(r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json') if r['prefix']=='model.layers.0.mlp.gate_up_proj');assert result['weight_signature']==expected
    plan={(r['case'],r['M']):r['configuration'] for r in read(BASE.parent/'tianshu_dense_w8a8_20261020/MODEL_INTEGRATION_PLAN_32g.json')['selected_configurations']}
    checks=0;summary=[];artifacts=[];failures=[]
    with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as checkpoint:
      for case in result['cases']:
        m,tile=case['M'],case['tile_id'];assert case['NN_configuration']==tiles[tile] and case['NT_configuration']==plan[('gateup',m)]
        rounds=[r for r in result['rounds'] if (r['M'],r['tile_id'])==(m,tile)]
        if case.get('compile_failed'):
            assert not case['complete'] and not rounds and not case['phases'] and case['exception_type'] in ('OutOfResources','CompilationError');failures.append(case);continue
        assert case['complete'] and case['timing_guards_passed'] and len(case['phases'])==3 and len(rounds)==7
        assert [r['round'] for r in rounds]==list(range(7))
        assert all(r['replays']==40 and r['order']==(['NT','NN'] if r['round']%2==0 else ['NN','NT']) and all(math.isfinite(r[k]) and r[k]>0 for k in ('NT','NN')) for r in rounds)
        for phase in case['phases']:
            assert all(phase[k] for k in ('full_INT32_bitwise','full_BF16_bitwise','all_A_quantization','CPU_selected_INT64','full_weights_immutable','guards','RNG_unchanged'))
            path=root/phase['artifact'];data=torch.load(path,map_location='cpu',weights_only=True);assert data['phase']==phase['phase']
            assert data['a'].shape==(m,2048)
            def row_for(i):return checkpoint.get_slice('model.layers.0.mlp.'+('gate_proj' if i<6144 else 'up_proj')+'.weight')[i%6144:i%6144+1]
            original=torch.cat([row_for(int(i)) for i in data['columns']],0);assert torch.equal(original,data['original_W_rows'])
            value=data['a'].float();asc=value.abs().amax(1).clamp_min(1.e-8)/127.;aq=torch.round(value/asc[:,None]).clamp(-127,127).to(torch.int8)
            assert torch.equal(aq,data['q']) and torch.equal(asc,data['ascale'])
            value=original.float();ws=value.abs().amax(1).clamp_min(1.e-8)/127.;wq=torch.round(value/ws[:,None]).clamp(-127,127).to(torch.int8)
            sign=1 if phase['phase']==0 else -1;factor=.5 if phase['phase']==2 else 1.
            exact=aq[data['rows']].to(torch.int64)@(wq*sign).to(torch.int64).T;assert torch.equal(exact,data['actual_GPU_integer'].to(torch.int64))
            reference=exact.float()*asc[data['rows'],None]*(ws*factor)[None,:];assert not bool(((data['actual_GPU_BF16'].float()-reference).abs()>(.02+.01*reference.abs())).any())
            checks+=1;artifacts.append(dict(file=phase['artifact'],bytes=path.stat().st_size,sha256=sha(path)))
        nt=statistics.median(r['NT'] for r in rounds);nn=statistics.median(r['NN'] for r in rounds)
        summary.append(dict(M=m,tile_id=tile,NT_ms=nt,NN_ms=nn,speedup_percent=100*(nt/nn-1),NT_configuration=case['NT_configuration'],NN_configuration=case['NN_configuration'],compiled=case['compiled']))
    assert checks==3*len(summary) and len(result['rounds'])==7*len(summary)
    best=[max((r for r in summary if r['M']==m),key=lambda r:r['speedup_percent']) for m in (8,64)]
    gate=dict(complete=True,independently_CPU_rechecked=True,model_executed=False,official_benchmark=False,Math_not_run=True,adopted=False,
        cases=len(result['cases']),passed_cases=len(summary),compile_failures=failures,full_device_INT32_BF16_checks=checks,CPU_selected_actual_GPU_INT64_checks=checks,
        source_sha256=sha(__file__),frozen_sources=result['frozen_sources'],raw_root=str(root),raw_result_sha256=sha(root/'result.json'),artifacts=artifacts,summary=summary,best_by_M=best,
        selection_scope='Exploratory minimum of 17 fixed candidates, not independent holdout or model score; original selected NT tile remains the common control; includes A quantize, excludes static W transpose.')
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'ROUNDS_32g.json').write_text(json.dumps(result['rounds'],indent=2)+'\n')
    with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as file:
        keys=('M','tile_id','NT_ms','NN_ms','speedup_percent','NN_configuration');writer=csv.DictWriter(file,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in summary)
    print(json.dumps({k:v for k,v in gate.items() if k not in ('frozen_sources','artifacts','summary','compile_failures')}));print(json.dumps(best))
if __name__=='__main__':main()
