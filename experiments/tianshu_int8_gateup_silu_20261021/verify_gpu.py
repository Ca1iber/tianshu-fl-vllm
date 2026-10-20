# SPDX-License-Identifier: Apache-2.0
"""Independently recompute saved A/W quantization, INT64 dots and activation."""
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pack(x):
    xf=x.float();scale=xf.abs().amax(-1).clamp_min(1.e-8)/127.
    return (xf/scale[...,None]).round().clamp(-127,127).to(torch.int8),scale

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve();state=read(root/'result.json')
    assert state['phase']=='complete' and state['complete'] and len(state['cases'])==24
    for path,digest in state['frozen_sources'].items():assert sha(path)==digest and sha(root/'source_snapshot'/Path(path).relative_to('/'))==digest
    rows=[];files={};phases=0;integer_values=0
    for case in state['cases']:
        assert len(case['phases'])==len(case['phase_files'])==3 and case['timing_guards_passed']
        for record,name in zip(case['phases'],case['phase_files']):
            assert all(record[key] for key in ('full_actual_timed_BF16_output_bitwise','full_diagnostic_INT32_and_projected_BF16_bitwise','full_quantization_exact','CPU_selected_INT64_and_projection_exact','CPU_selected_activation_tolerance_passed','all_guards_inputs_RNG_preserved','padded_output_zero'))
            file=root/name;data=torch.load(file,map_location='cpu',weights_only=True)
            aq,asc=pack(data['A']);wq,wsc=pack(data['original_W_rows'])
            assert torch.equal(aq,data['A_INT8']) and torch.equal(asc,data['A_scale'])
            assert torch.equal(wq*data['weight_sign'],data['W_INT8']) and torch.equal(wsc*data['weight_scale_factor'],data['W_scale'])
            dot=aq[data['rows']].to(torch.int64)@data['W_INT8'].to(torch.int64).T
            assert torch.equal(dot,data['native_integer'].to(torch.int64)) and torch.equal(dot,data['fused_integer'].to(torch.int64))
            projection=(dot.float()*asc[data['rows'],None]*data['W_scale'][None,:]).to(torch.bfloat16)
            assert torch.equal(projection.view(torch.int16),data['native_projected'].view(torch.int16)) and torch.equal(projection.view(torch.int16),data['fused_projected'].view(torch.int16))
            gate,up=projection[:,:16].float(),projection[:,16:].float();expected=gate/(1.+torch.exp(-gate))*up
            actual=data['native_output'];assert torch.equal(actual.view(torch.int16),data['fused_output'].view(torch.int16))
            assert torch.isfinite(actual).all() and ((actual.float()-expected).abs()<=.001+.02*expected.abs()).all()
            files[name]=dict(sha256=sha(file),bytes=file.stat().st_size);phases+=1;integer_values+=dot.numel()
        rounds=[r for r in state['timing_rounds'] if r['M']==case['M'] and r['strided']==case['strided'] and r['configuration']==case['configuration']]
        assert len(rounds)==7 and [r['round'] for r in rounds]==list(range(7)) and all(r['replays']==20 for r in rounds)
        native=statistics.median(r['native_ms'] for r in rounds);fused=statistics.median(r['fused_ms'] for r in rounds)
        assert native>0 and fused>0
        rows.append(dict(M=case['M'],padded_M=case['padded_M'],strided=case['strided'],configuration=case['configuration'],
            native_ms=native,fused_ms=fused,speedup_percent=100*(native/fused-1),actual_timed_compiled=case['compiled_fused_actual_timed'],diagnostic_compiled=case['compiled_fused_diagnostic']))
    result=dict(complete=True,independently_CPU_rechecked=True,component_only=True,adopted=False,model_quality_not_evaluated=True,
        full_actual_timed_outputs_bitwise_verified=True,checked_configurations=24,changed_input_Graph_phases=phases,CPU_INT64_values=integer_values,
        quantization_and_native_BF16_projection_rounds_verified=True,all_RNG_inputs_guards_preserved=True,
        timing_includes_activation_quantizer=True,weight_packing_excluded_from_both_arms=True,comparison_to_best_qualified_native_integer_plus_actual_SiLU=True,
        raw_root=str(root),raw_result_sha256=sha(root/'result.json'),frozen_sources=state['frozen_sources'],fixture_files=files,rows=rows,verifier_sha256=sha(__file__))
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');(BASE/'ROUNDS_32g.json').write_text(json.dumps(state['timing_rounds'],indent=2)+'\n')
    print(json.dumps(dict(complete=True,configurations=24,phases=phases,rows=rows)))
if __name__=='__main__':main()
