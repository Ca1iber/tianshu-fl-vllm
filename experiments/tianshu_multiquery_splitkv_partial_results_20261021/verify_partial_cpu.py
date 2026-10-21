# SPDX-License-Identifier: Apache-2.0
"""Offline CPU recheck of frozen evidence, all timings, selected original-K refs."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent
from source import sha,CONFIGS
p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=args.raw;out=args.output;out.mkdir(exist_ok=False)
result=json.loads((root/'result.json').read_text());assert result['complete'] is False and result['phase']=='failed' and result['error'].startswith("AssertionError(('q2_short_k2', 'control_2D', 0,")
assert not result['model_executed'] and not result['official_benchmark'] and result['numerical_JIT_AST_unchanged']
result["cases"]=[c for c in result["cases"] if len(c["timings"])==7]
for path,h in result['frozen_sources'].items():
    assert sha(path)==h and sha(root/'source_snapshot'/Path(path).relative_to('/'))==h
import torch
torch.set_num_threads(8)
assert len(result['cases'])==19 and len(result['timing_rounds'])==19*7*7
cpu_vectors=0;codec_values=0;rows=[]
for case in result['cases']:
    assert len(case['phases'])==2 and case['timed_full_KV_inputs_RNG_preserved']
    assert sha(root/(case['name']+'.pt'))==case['tensor_archive_sha256']
    data=torch.load(root/(case['name']+'.pt'),map_location='cpu',weights_only=True)
    assert len(data['phases'])==2 and data['query_lengths']==case['query_lengths']
    for phase,raw_phase in zip(case['phases'],data['phases'],strict=True):
        assert phase['phase']==raw_phase['phase'] and len(phase['full_GPU_checks'])==7 and all(c['passed'] for c in phase['CPU_checks'])
        expected=raw_phase['expected'];assert expected.dtype==torch.float32
        for c in phase['full_GPU_checks']:
            assert c['passed'] and c['guards_preserved'] and c['input_KV_and_metadata_unchanged'] and c['RNG_unchanged']
            prod=c['actual_producer'];assert prod['mode']==('control' if c['label']=='control_2D' else 'candidate')
            assert prod['IS_3D']==(prod['mode']=='candidate') and prod['query_tokens']==sum(case['query_lengths'])
            if prod['mode']=='candidate':
                index=int(c['label'].rsplit('_',1)[1]);assert prod['config']==CONFIGS[index] and c['actual_reducer']
                assert prod['grid']==[prod['query_tokens']//prod['block_q']+case['sequences'],2,prod['segments']]
            else:assert prod['config']==dict(block_m=256,tile=128,warps=16,stages=1)
            actual=raw_phase['outputs'][c['label']].float();assert actual.shape==expected.shape and bool(torch.isfinite(actual).all())
            assert not bool(((actual-expected).abs()>(.002+.01*expected.abs())).any())
        for seq,original in raw_phase['selected_original_K'].items():
            seq=int(seq);bits=original.view(torch.int16).int();mag=bits&32767;rounded=((mag+7+((mag>>4)&1))>>4)-928
            code=torch.where(rounded>=8,rounded.clamp(max=127),0)|((bits&32768)>>8);magnitude=code&127
            decoded=(torch.where(magnitude>=8,(magnitude<<4)+14848,0)|((code&128)<<8)).to(torch.int16).view(torch.bfloat16)
            assert torch.equal(decoded,raw_phase['selected_K'][seq]);codec_values+=decoded.numel()
        for sel in raw_phase['selected']:
            seq,pos,head,token=(sel[k] for k in ('seq','pos','head','token'));ql=case['query_lengths'][seq];kl=raw_phase['key_lengths'][seq]
            end=kl-ql+pos+1 if case['causal'] else kl
            k=raw_phase['selected_K'][seq][:end,head//8].float();v=raw_phase['selected_V'][seq][:end,head//8].float();q=raw_phase['q'][token,head].float()
            answer=((k@q)*128**-.5).softmax(0)@v;assert not bool(((expected[token,head]-answer).abs()>(.002+.01*answer.abs())).any());cpu_vectors+=1
    baseline=case['timings']['control_2D']['median_ms']
    for label,t in case['timings'].items():
        r=[r for r in result['timing_rounds'] if r['case']==case['name'] and r['label']==label]
        assert len(r)==7 and {x['round'] for x in r}==set(range(7)) and all(x['replays']==20 and x['ms']>0 for x in r)
        assert [x['ms'] for x in r]==t['all_ms'] and statistics.median(t['all_ms'])==t['median_ms']
        rows.append(dict(case=case['name'],sequences=case['sequences'],max_query=max(case['query_lengths']),max_key=max(case['key_lengths']),label=label,
            median_ms=t['median_ms'],relative_to_direct_2D_percent=(baseline/t['median_ms']-1)*100,prefill_arena_fits=case['full_prefill_arena_fits_8192_pages']))
summary=dict(complete=True,component_plan_complete=False,original_GPU_plan_failed=True,scope="Only the 19 completed long-context fixtures; failed short control and remaining six fixtures are unverified.",CPU_reverified=True,model_executed=False,official_benchmark=False,adopted=False,
    raw_root=str(root),raw_result_sha256=sha(root/'result.json'),source_count=result['source_count'],frozen_sources=result['frozen_sources'],
    cases=19,full_GPU_phase_modes=19*2*7,offline_full_saved_output_comparisons=19*2*7,CPU_attention_vectors=cpu_vectors,
    CPU_original_K_codec_values=codec_values,actual_2D_and_3D_launches_verified=True,timing_trials=19*7*7,
    timing_scope=result['timing_scope'],baseline_scope=result['baseline_scope'],six_configs_are_exploratory_not_holdout=True,
    verifier_sha256=sha(Path(__file__)),rows=rows)
(out/'GPU_GATE_32g.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'KEY_TIMINGS_32g.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','frozen_sources')}))
