# SPDX-License-Identifier: Apache-2.0
"""CPU offline output/codec/reference/real-route/timing closure verification."""
import argparse,csv,importlib.util,json,statistics
from pathlib import Path
from factory import sha,CONFIGS,LABELS,EMULATE
p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=args.raw;out=args.output;out.mkdir(exist_ok=False)
r=json.loads((root/'result.json').read_text());assert r['complete'] and r['phase']=='complete' and len(r['cases'])==26
assert not r['model_executed'] and not r['official_benchmark'] and r['candidate_numerical_JIT_AST_unchanged']
for path,h in r['frozen_sources'].items():assert sha(path)==h and sha(root/'source_snapshot'/Path(path).relative_to('/'))==h
import torch
torch.set_num_threads(8)
spec=importlib.util.spec_from_file_location('_paths_offline_emulation',EMULATE);emu=importlib.util.module_from_spec(spec);spec.loader.exec_module(emu)
configs=dict(zip(LABELS,CONFIGS,strict=True));assert len(r['timing_rounds'])==26*5*7
full=0;vectors=0;codec_values=0;short_full_recomputed=0;ideal_failed=[];rows=[]
def encode_decode(original):
    a=original.view(torch.int16).int();m=a&32767;z=((m+7+((m>>4)&1))>>4)-928
    c=torch.where(z>=8,z.clamp(max=127),0)|((a&32768)>>8);mag=c&127
    return (torch.where(mag>=8,(mag<<4)+14848,0)|((c&128)<<8)).to(torch.int16).view(torch.bfloat16)
for case in r['cases']:
    assert len(case['phases'])==2 and case['timed_full_KV_inputs_RNG_preserved'] and case['timed_native_arena_guard_and_ids_preserved']
    assert sha(root/(case['name']+'.pt'))==case['tensor_archive_sha256']
    a=torch.load(root/(case['name']+'.pt'),map_location='cpu',weights_only=True)
    assert len(a['phases'])==2 and a['query_lengths']==case['query_lengths']
    for phase,data in zip(case['phases'],a['phases'],strict=True):
        assert phase['phase']==data['phase'] and len(phase['full_GPU_checks'])==5 and all(x['passed'] for x in phase['CPU_checks'])
        for seq,original in data['selected_original_K'].items():
            decoded=encode_decode(original);assert torch.equal(decoded,data['selected_K'][int(seq)]);codec_values+=decoded.numel()
        for c in phase['full_GPU_checks']:
            assert c['passed'] and c['guards_preserved'] and c['RNG_unchanged'] and c['input_KV_and_metadata_unchanged'] and c['native_arena_guard_and_ids_preserved']
            label=c['label'];prod=c['actual_producer'];assert prod['query_tokens']==sum(case['query_lengths'])
            if label=='bounded_gather_or_byte':
                expected_path='actual_bounded_gather_plus_native_BF16' if case['causal'] and case['full_prefill_arena_fits_8192_pages'] else 'direct_byte_fallback'
                assert prod['actual_route']==expected_path and prod['arena_capacity_pages']==8192 and not prod['IS_3D']
            else:
                assert prod['config']==configs[label] and prod['IS_3D'] and c['actual_reducer']
                assert prod['grid']==[sum(case['query_lengths'])//prod['block_q']+case['sequences'],2,prod['segments']]
            actual=data['outputs'][label].float()
            if c['reference_kind']=='independent_CPU_native_BF16_P':
                fixture=data['short_fixture'];assert fixture is not None
                decoded=encode_decode(fixture['original_K']);assert torch.equal(decoded,fixture['decoded_K'])
                cfg=configs[label] if label in configs else dict(block_m=256,tile=128,warps=16,stages=1)
                # All short cases are N8, matching S16 in the true native policy.
                reference=emu.emulate(data['q'],decoded,fixture['V'],case['query_lengths'],data['key_lengths'],cfg,label in configs,16,case['causal']).float()
                assert torch.equal(reference,data['native_BF16_P_references'][label].float());short_full_recomputed+=1
                d=(actual-data['expected']).abs();bad=d>(.002+.01*data['expected'].abs())
                assert int(bad.sum())==c['ideal_FP32_diagnostic']['failed']
                if bool(bad.any()):ideal_failed.append(dict(case=case['name'],phase=phase['phase'],label=label,failed=int(bad.sum()),max_abs=float(d.max())))
            else:
                assert c['reference_kind']=='ideal_FP32';reference=data['expected']
            assert actual.shape==reference.shape and bool(torch.isfinite(actual).all()) and not bool(((actual-reference).abs()>(.002+.01*reference.abs())).any());full+=1
        for select in data['selected']:
            seq,pos,h,t=(select[k] for k in ('seq','pos','head','token'));ql=case['query_lengths'][seq];kl=data['key_lengths'][seq];end=kl-ql+pos+1 if case['causal'] else kl
            k=data['selected_K'][seq][:end,h//8].float();v=data['selected_V'][seq][:end,h//8].float();q=data['q'][t,h].float()
            reference=((k@q)*128**-.5).softmax(0)@v
            assert not bool(((data['expected'][t,h]-reference).abs()>(.002+.01*reference.abs())).any());vectors+=1
    baseline=case['timings']['bounded_gather_or_byte']['median_ms'];old3d=case['timings'][LABELS[0]]['median_ms']
    for label,timing in case['timings'].items():
        times=[x for x in r['timing_rounds'] if x['case']==case['name'] and x['label']==label]
        assert len(times)==7 and {x['round'] for x in times}==set(range(7)) and all(x['replays']==20 and x['ms']>0 for x in times)
        assert [x['ms'] for x in times]==timing['all_ms'] and statistics.median(timing['all_ms'])==timing['median_ms']
        rows.append(dict(case=case['name'],sequences=case['sequences'],max_query=max(case['query_lengths']),max_key=max(case['key_lengths']),label=label,
            median_ms=timing['median_ms'],vs_bounded_path_percent=(baseline/timing['median_ms']-1)*100,vs_old3D_percent=(old3d/timing['median_ms']-1)*100,
            matched_uniform_target_or_fallback=case['baseline_matches_uniform_or_fallback_path'],prefill_arena_fits=case['full_prefill_arena_fits_8192_pages']))
summary=dict(complete=True,CPU_reverified=True,model_executed=False,official_benchmark=False,adopted=False,
    raw_root=str(root),raw_result_sha256=sha(root/'result.json'),source_count=r['source_count'],frozen_sources=r['frozen_sources'],cases=26,
    full_saved_output_comparisons=full,CPU_ideal_attention_vectors=vectors,CPU_original_K_codec_values=codec_values,CPU_full_native_BF16_P_recomputed=short_full_recomputed,
    ideal_FP32_short_failures_preserved=ideal_failed,original_ideal_tolerance_unchanged=True,short_native_semantics_and_ideal_diagnostics_distinguished=True,
    actual_bounded_gather_and_byte_fallback_verified=True,all_guards_and_RNG_verified=True,timing_trials=26*5*7,
    timing_scope=r['timing_scope'],baseline_scope=r['baseline_scope'],four_configs_are_exploratory_not_holdout=True,verifier_sha256=sha(__file__),rows=rows)
(out/'GPU_GATE_32g.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'KEY_TIMINGS_32g.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in summary.items() if k not in ('frozen_sources','rows','ideal_FP32_short_failures_preserved')}))
