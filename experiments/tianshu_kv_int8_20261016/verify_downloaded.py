# SPDX-License-Identifier: Apache-2.0
"""Recompute INT8 fixture micro records; quantized accuracy is a separate task."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve();state=read(root/'result.json')
    assert state['phase']=='complete' and state['complete'] and not state['official_benchmark'] and not state['adopted']
    assert state['manual_quantization_not_native_writer'] and state['full_model_and_Math_not_run']
    repo=Path(__file__).resolve().parents[2]
    for source,sha in state['source_snapshots'].items():
        assert digest(root/'source_snapshot'/Path(source).relative_to('/'))==sha
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
            assert digest(repo/Path(source).relative_to('/workspace/vllm-plugin-FL'))==sha
    variants={'BF16','INT8_packed128_w1','INT8_inline132_w1','INT8_packed128_w2'}
    onload='derivation' in state
    if onload:
        from dequant_on_load import render_source
        generated=root/'dequant_generated/generated_dequant_on_load.py'
        original=next(root.glob('decode_coherent_records/cache_generated/*/generated_cache_policy.py'))
        text,derivation=render_source(original.read_bytes())
        assert generated.read_text()==text
        for key,value in derivation.items():assert state['derivation'][key]==value
        variants={'BF16','INT8_packed128_original','INT8_packed128_load_w1','INT8_packed128_load_w2','INT8_inline132_load_w1'}
    assert len(state['cases'])==3
    assert {(c['batch'],c['context']) for c in state['cases']}=={(8,33),(64,4096),(64,5120)}
    times=[];all_approximations=[];events=0
    for case in state['cases']:
        n=case['batch'];assert len(case['lengths'])==n and max(case['lengths'])==case['context']
        results=case['results'];assert {r['name'] for r in results}==variants and len(results)==len(variants)
        baseline=next(r for r in results if r['name']=='BF16')['median_ms']
        for r in results:
            raw=r['raw_ms'];assert len(raw)==7 and all(t>0 for t in raw)
            assert statistics.median(raw)==r['median_ms'];events+=7
            assert abs(r['time_change_percent']-100*(r['median_ms']/baseline-1))<1e-12
            meta=r['metadata'];bf16=r['name']=='BF16'
            assert meta['grid']==[n//2+n,2,8 if n==64 else 16]
            assert meta['quant_mode']==(0 if bf16 else 2) and meta['key_dtype']==('torch.bfloat16' if bf16 else 'torch.int8')
            assert meta['compiled_warps']==meta['warps']==(2 if r['name'].endswith('w2') else 1)
            assert meta['tile']==(32 if r['name'].endswith('w2') else 16)
            if onload:assert meta['onload_dequant']==('_load_' in r['name'])
            if 'inline132' in r['name']:
                assert r['key_strides']==[8448,264,132,1] and r['scale_strides']==[2112,66,33]
            else:
                assert r['key_strides']==[4096,256,128,1]
                assert r['scale_strides']==(None if bf16 else [32,2,1])
            times.append(dict(batch=n,context=case['context'],name=r['name'],median_ms=r['median_ms'],time_change_percent=r['time_change_percent'],metadata=meta))
        assert len(case['phases'])==2
        for phase in case['phases']:
            assert phase['outputs_cleared_before_replay']
            for key in ('CPU_BF16_reference','CPU_dequantized_reference'):
                assert phase[key] and all(c['passed'] and c['failed_elements']==0 for c in phase[key])
            assert {r['name'] for r in phase['checks']}==variants
            for r in phase['checks']:
                check=r['against_own_FP32'];assert check['passed'] and check['failed_elements']==0 and check['atol']==.02 and check['rtol']==.01 and r['tails_preserved']
                approximation=r['against_BF16_FP32'];assert all(math.isfinite(v) and v>=0 for v in approximation.values())
                all_approximations.append(dict(batch=n,context=case['context'],phase=phase['phase'],name=r['name'],**approximation))
    report=dict(complete=True,all_source_snapshots_and_owned_current_sources_checked=len(state['source_snapshots']),
        three_cases_and_six_changed_input_phase_records_checked=True,all_raw_events_recomputed=events,
        quantized_FP32_reference_separate_from_BF16_approximation=True,
        full_GPU_tensor_values_not_serialized=True,timings=times,approximation_metrics=all_approximations,
        dequant_on_load_variant=onload,native_KV_writer_and_quantization_costs_excluded=True,
        service_and_Math_not_run=True,official_benchmark=False,adopted=False)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='approximation_metrics'}))


if __name__=='__main__':main()
