# SPDX-License-Identifier: Apache-2.0
"""CPU recheck checkpoint rows, quantization, INT64 dots, full saved outputs."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
import torch
from safetensors import safe_open
from choices import CASES,configurations,baseline
BASE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pack(x):
    xf=x.float();scale=xf.abs().amax(-1).clamp_min(1.e-8)/127.
    return (xf/scale[...,None]).round().clamp(-127,127).to(torch.int8),scale

def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root.resolve();state=read(root/'result.json')
    assert state['phase']=='complete' and state['complete'] and len(state['cases'])==32
    expected={(m,s,json.dumps(c,sort_keys=True)) for m,s in CASES for c in configurations(m)}
    assert {(r['M'],r['strided'],json.dumps(r['configuration'],sort_keys=True)) for r in state['cases']}==expected
    for p,h in state['frozen_sources'].items():assert sha(p)==h==sha(root/'source_snapshot'/Path(p).relative_to('/'))
    reference=next(r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json') if r['prefix']=='model.layers.0.mlp.gate_up_proj')
    assert all(state['weight_signature'][k]==reference[k] for k in ('original_BF16_sha256','packed_sha256','scales_sha256'))
    rows=[];files={};failures=[];phases=integers=elements=0
    with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as checkpoint:
      for case in state['cases']:
        m=case['M'];assert case['padded_M']==m and case['baseline_configuration']==baseline(m)
        rounds=[r for r in state['timing_rounds'] if r['M']==m and r['strided']==case['strided'] and r['configuration']==case['configuration']]
        if case.get('compile_failed'):
            assert case['exception_type'] in ('OutOfResources','CompilationError') and not case['complete'] and not case['phases'] and not rounds
            failures.append(case);continue
        assert case['complete'] and case['timing_guards_passed'] and len(case['phases'])==len(case['phase_files'])==3
        for record,name in zip(case['phases'],case['phase_files']):
            assert all(record[k] for k in ('full_actual_timed_BF16_output_bitwise','full_diagnostic_INT32_and_projected_BF16_bitwise','full_quantization_exact','CPU_selected_INT64_and_projection_exact','CPU_selected_activation_tolerance_passed','all_guards_inputs_RNG_preserved','no_actual_M_padding'))
            path=root/name;d=torch.load(path,map_location='cpu',weights_only=True)
            assert d['A'].shape==(m,2048) and d['phase']==record['phase']
            def original_row(i):return checkpoint.get_slice('model.layers.0.mlp.'+('gate_proj' if i<6144 else 'up_proj')+'.weight')[i%6144:i%6144+1]
            original=torch.cat([original_row(int(c)) for c in d['columns']],0);assert torch.equal(original,d['original_W_rows'])
            aq,asc=pack(d['A']);wq,ws=pack(original)
            assert torch.equal(aq,d['A_INT8']) and torch.equal(asc,d['A_scale'])
            assert torch.equal(wq*d['weight_sign'],d['W_INT8']) and torch.equal(ws*d['weight_scale_factor'],d['W_scale'])
            dot=aq[d['rows']].to(torch.int64)@d['W_INT8'].to(torch.int64).T
            assert torch.equal(dot,d['native_integer'].to(torch.int64)) and torch.equal(dot,d['fused_integer'].to(torch.int64))
            projection=(dot.float()*asc[d['rows'],None]*d['W_scale'][None,:]).to(torch.bfloat16)
            assert torch.equal(projection.view(torch.int16),d['native_projected'].view(torch.int16)) and torch.equal(projection.view(torch.int16),d['fused_projected'].view(torch.int16))
            gate,up=projection[:,:16].float(),projection[:,16:].float();answer=gate/(1.+torch.exp(-gate))*up
            actual=d['native_output'];assert torch.equal(actual.view(torch.int16),d['fused_output'].view(torch.int16))
            assert torch.isfinite(actual).all() and ((actual.float()-answer).abs()<=.001+.02*answer.abs()).all()
            full_a,full_b=d['full_timed_native_output'],d['full_timed_fused_output']
            assert full_a.shape==full_b.shape==(m,6144) and full_a.dtype==full_b.dtype==torch.bfloat16
            assert torch.equal(full_a.view(torch.int16),full_b.view(torch.int16)) and torch.isfinite(full_a).all()
            assert torch.equal(full_a[d['rows'][:,None],d['columns'][:16][None,:]],actual)
            files[name]=dict(bytes=path.stat().st_size,sha256=sha(path));phases+=1;integers+=dot.numel();elements+=full_a.numel()
        assert len(rounds)==7 and [r['round'] for r in rounds]==list(range(7))
        assert all(r['replays']==20 and r['order']==(['native','fused'] if r['round']%2==0 else ['fused','native']) and all(math.isfinite(r[k]) and r[k]>0 for k in ('native_ms','fused_ms')) for r in rounds)
        native=statistics.median(r['native_ms'] for r in rounds);fused=statistics.median(r['fused_ms'] for r in rounds)
        rows.append(dict(M=m,strided=case['strided'],configuration=case['configuration'],native_ms=native,fused_ms=fused,speedup_percent=100*(native/fused-1),actual_timed_compiled=case['compiled_fused_actual_timed'],diagnostic_compiled=case['compiled_fused_diagnostic']))
    assert len(rows)+len(failures)==32 and len(state['timing_rounds'])==7*len(rows) and phases==3*len(rows)
    best=[max((r for r in rows if (r['M'],r['strided'])==(m,s)),key=lambda r:r['speedup_percent']) for m,s in CASES if any((r['M'],r['strided'])==(m,s) for r in rows)]
    assert not torch.cuda.is_initialized()
    result=dict(complete=True,independently_CPU_rechecked=True,CUDA_not_initialized=True,component_only=True,adopted=False,model_quality_not_evaluated=True,
        full_actual_timed_outputs_bitwise_verified=True,checked_configurations=len(rows),compile_failures=failures,changed_input_Graph_phases=phases,CPU_INT64_values=integers,full_timed_BF16_elements_reloaded=elements,
        actual_checkpoint_weight_rows_verified=True,no_actual_M_padding=True,current_selected_NN_GEMM_configuration_used=True,
        quantization_and_native_BF16_projection_rounds_verified=True,all_RNG_inputs_guards_preserved=True,timing_includes_activation_quantizer=True,weight_packing_excluded_from_both_arms=True,
        comparison_to_NN_integer_plus_declared_FlagGems_SiLU=True,qualified_compiled_model_activation_equivalence_not_asserted=True,
        raw_root=str(root),raw_result_sha256=sha(root/'result.json'),frozen_sources=state['frozen_sources'],fixture_files=files,rows=rows,best_by_shape=best,verifier_sha256=sha(__file__))
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');(BASE/'ROUNDS_32g.json').write_text(json.dumps(state['timing_rounds'],indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('frozen_sources','fixture_files','rows','compile_failures')}))
if __name__=='__main__':main()
