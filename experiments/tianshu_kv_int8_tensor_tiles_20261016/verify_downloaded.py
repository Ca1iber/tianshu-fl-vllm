# SPDX-License-Identifier: Apache-2.0
"""Audit page-interleaved INT8 launch screening records, not model quality."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tianshu_kv_int8_tensor_20261016'))
from fold_value_scale import render_source


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root.resolve()
    state=json.loads((root/'result.json').read_text())
    assert state['phase']=='complete' and state['complete'] and not state['adopted'] and not state['official_benchmark']
    assert state['manual_quantization_not_native_writer'] and state['full_model_and_Math_not_run']
    repo=Path(__file__).resolve().parents[2]
    for name,sha in state['source_snapshots'].items():
        assert digest(root/'source_snapshot'/Path(name).relative_to('/'))==sha,name
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert digest(repo/Path(name).relative_to('/workspace/vllm-plugin-FL'))==sha,name
    original=next(root.glob('decode_coherent_records/cache_generated/*/generated_cache_policy.py'))
    generated=root/'value_scale_generated/generated_value_scale.py'
    text,derivation=render_source(original.read_bytes());assert generated.read_text()==text
    for k,v in derivation.items():assert state['derivation'][k]==v
    def producer(text):return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_unified_attention')
    assert ast.dump(producer(original.read_text()))==ast.dump(producer(text))
    plans={'BF16':(16,1,8),**{f'INT8_t{t}_w{w}_s{s}':(t,w,s) for t,w,s in ((16,1,8),(16,2,8),(16,4,8),(16,8,8),(32,4,8),(16,2,16),(16,4,16),(16,2,4))}}
    assert len(state['cases'])==2
    assert {(c['batch'],c['context']) for c in state['cases']}=={(64,4096),(64,5120)}
    timings=[];approximations=[];events=checks=0
    for c in state['cases']:
        assert c['lengths']==[c['context']-i*37%257 for i in range(64)]
        assert not c['compile_failures'], 'All planned variants compiled in this recorded run'
        assert len(c['quantization'])==2
        for q in c['quantization']:
            assert q['K_scale']==q['V_scale']==1/16
            assert q['elements_per_K_or_V']==64*((c['context']+15)//16)*16*2*128
            assert q['K_clipped']==q['V_clipped']==0
        rows=c['results'];assert len(rows)==len(plans) and {r['name'] for r in rows}==plans.keys()
        baseline=next(r['median_ms'] for r in rows if r['name']=='BF16')
        for r in rows:
            t,w,s=plans[r['name']];m=r['metadata'];quant=r['name']!='BF16'
            assert len(r['raw_ms'])==7 and all(math.isfinite(v) and v>0 for v in r['raw_ms'])
            assert statistics.median(r['raw_ms'])==r['median_ms']
            assert abs(r['time_change_percent']-100*(r['median_ms']/baseline-1))<1e-12
            assert m['grid']==[96,2,s] and m['tile']==t and m['warps']==m['compiled_warps']==w
            assert m['quant_mode']==0 and m['key_dtype']==('torch.int8' if quant else 'torch.bfloat16')
            assert m['n_regs']>0 and m['n_spills']==0
            assert r['segments']==s and r['scale_folded']==quant and r['key_strides']==[8192,256,128,1]
            timings.append(dict(batch=64,context=c['context'],**r));events+=7
        assert [p['phase'] for p in c['phases']]==[0,1]
        for p in c['phases']:
            assert p['outputs_cleared_before_replay']
            for key in ('CPU_BF16_reference','CPU_dequantized_reference'):
                assert len(p[key])==12 and all(x['passed'] and x['failed_elements']==0 for x in p[key])
            assert len(p['checks'])==len(plans) and {x['name'] for x in p['checks']}==plans.keys()
            for x in p['checks']:
                a=x['against_own_FP32'];assert a['passed'] and a['failed_elements']==0 and a['atol']==.02 and a['rtol']==.01
                assert x['tails_preserved'] and all(math.isfinite(v) and v>=0 for v in x['against_BF16_FP32'].values())
                approximations.append(dict(context=c['context'],phase=p['phase'],name=x['name'],**x['against_BF16_FP32']));checks+=1
    audit=dict(complete=True,source_snapshots_verified=len(state['source_snapshots']),all126_events_recomputed=events==126,
        four_changed_input_phases_checked=True,own_reference_checks=checks,all8_quantized_launch_variants_slower=True,
        producer_AST_unchanged=True,generated_reducer_reconstructed=True,zero_clipping_fixture_only=True,
        full_GPU_tensor_values_not_serialized=True,manual_page_interleaved_storage_not_full_allocator=True,
        no_native_writer_service_or_Math=True,INT8_storage_not_FP8=True,adopted=False,timings=timings,approximations=approximations)
    assert events==126 and checks==36 and all(r['time_change_percent']>0 for r in timings if r['name']!='BF16')
    (root/'local_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('timings','approximations')}))


if __name__=='__main__':main()
