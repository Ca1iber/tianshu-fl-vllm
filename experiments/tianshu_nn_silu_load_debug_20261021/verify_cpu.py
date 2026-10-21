# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261021_nn_silu_load_debug_GPU_tianshu')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    r=json.loads((ROOT/'result.json').read_text());assert r['complete'] and r['diagnostic_only'] and not r['adopted'] and sha(ROOT/'stages.pt')==r['payload_sha256']
    for p,h in r['sources'].items():assert sha(p)==h==sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))
    d=torch.load(ROOT/'stages.pt',map_location='cpu',weights_only=True);a=d['A'].float();s=a.abs().amax(1).clamp_min(1.e-8)/127.;q=(a/s[:,None]).round().clamp(-127,127).to(torch.int8)
    assert torch.equal(q,d['Q']) and torch.equal(s,d['A_scale'])
    assert hashlib.sha256(memoryview(d['W_NN'].T.contiguous().view(torch.uint8).numpy())).hexdigest()==r['weight_signature']['packed_sha256']
    assert hashlib.sha256(memoryview(d['W_scale'].view(torch.uint8).numpy())).hexdigest()==r['weight_signature']['scales_sha256']
    dot=q.long()@d['W_NN'].long();projection=(dot.float()*s[:,None]*d['W_scale'][None,:]).to(torch.bfloat16)
    assert torch.equal(dot,d['native_integer'].long()) and torch.equal(projection.view(torch.int16),d['native_projected'].view(torch.int16))
    result={}
    for mode,v in d['variants'].items():
        counts=dict(integer_vs_full_CPU=int((v['diagnostic_integer'].long()!=dot).sum()),projection_vs_full_CPU=int((v['diagnostic_projected'].view(torch.int16)!=projection.view(torch.int16)).sum()),
                    actual_output_vs_native=int((v['actual_timed'].view(torch.int16)!=d['native_timed'].view(torch.int16)).sum()),diagnostic_output_vs_native=int((v['diagnostic_output'].view(torch.int16)!=d['native_timed'].view(torch.int16)).sum()))
        assert counts['integer_vs_full_CPU']==r['variants'][mode]['integer_differences'] and counts['actual_output_vs_native']==r['variants'][mode]['actual_output_differences']
        result[mode]=dict(**counts,all_stage_bitwise=not any(counts.values()),compiled=r['variants'][mode])
    assert not torch.cuda.is_initialized()
    gate=dict(complete=True,independent_CPU_rechecked=True,diagnostic_only=True,adopted=False,original_bad_case_not_overwritten=True,CPU_INT64_elements=dot.numel(),variants=result,raw_root=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),verifier_sha256=sha(__file__))
    (BASE/'LOAD_DEBUG_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
