# SPDX-License-Identifier: Apache-2.0
"""Full CPU INT64 result and activation errors for the saved failing case."""
import hashlib,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261021_nn_silu_single_debug_GPU_tianshu')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    r=json.loads((ROOT/'result.json').read_text());assert r['complete'] and r['diagnostic_only'] and not r['adopted']
    assert sha(ROOT/'stages.pt')==r['payload_sha256']
    for p,h in r['sources'].items():assert sha(p)==h==sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))
    d=torch.load(ROOT/'stages.pt',map_location='cpu',weights_only=True);a=d['A'].float();scale=a.abs().amax(1).clamp_min(1.e-8)/127.;q=(a/scale[:,None]).round().clamp(-127,127).to(torch.int8)
    assert torch.equal(q,d['Q']) and torch.equal(scale,d['A_scale'])
    w=d['W_NN'].T.contiguous();assert hashlib.sha256(memoryview(w.view(torch.uint8).numpy())).hexdigest()==r['weight_signature']['packed_sha256']
    assert hashlib.sha256(memoryview(d['W_scale'].view(torch.uint8).numpy())).hexdigest()==r['weight_signature']['scales_sha256']
    dot=q.to(torch.int64)@d['W_NN'].to(torch.int64)
    projection=(dot.float()*scale[:,None]*d['W_scale'][None,:]).to(torch.bfloat16)
    gate,up=projection[:,:6144].float(),projection[:,6144:].float();answer=gate/(1+torch.exp(-gate))*up
    def errors(actual,expected):
        actual=actual.float();expected=expected.float();delta=(actual-expected).abs()
        return dict(differences=int((actual!=expected).sum()),max_abs=float(delta.max()),failed_old_CPU_tolerance=int((delta>(.001+.02*expected.abs())).sum()),finite=bool(torch.isfinite(actual).all()))
    checks=dict(CPU_INT64_elements=dot.numel(),native_integer_vs_CPU=int((d['native_integer'].long()!=dot).sum()),diagnostic_integer_vs_CPU=int((d['diagnostic_integer'].long()!=dot).sum()),native_projection_vs_CPU=errors(d['native_projected'],projection),diagnostic_projection_vs_CPU=errors(d['diagnostic_projected'],projection),
        actual_output_vs_native=errors(d['actual_timed'],d['native_timed']),diagnostic_output_vs_native=errors(d['diagnostic_output'],d['native_timed']),actual_output_vs_CPU=errors(d['actual_timed'],answer),native_output_vs_CPU=errors(d['native_timed'],answer),diagnostic_output_vs_CPU=errors(d['diagnostic_output'],answer))
    assert checks['native_integer_vs_CPU']==0
    assert not torch.cuda.is_initialized()
    report=dict(complete=True,independent_CPU_recheck=True,diagnostic_only=True,adopted=False,original_strict_gate_still_failed=True,actual_binary_matches_original_failed=r['actual_binary_matches_original_failed'],GPU_observed={k:r[k] for k in ('integer_difference_count','projection_difference_count','actual_output_difference_count','diagnostic_output_difference_count')},checks=checks,raw_root=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),verifier_sha256=sha(__file__))
    (BASE/'DEBUG_GATE_32g.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
