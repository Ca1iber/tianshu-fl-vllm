# SPDX-License-Identifier: Apache-2.0
"""Independently re-quantize saved source rows and classify the failed baseline."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import torch
    torch.set_num_threads(4)
    root=Path('/workspace/benchmark-results/20261020_dense_runtime_m_M64_diagnostic_tianshu');data=json.loads((root/'result.json').read_text());assert data['complete']
    for name,digest in data['frozen_sources'].items():assert sha(name)==sha(root/'source_snapshot'/Path(name).relative_to('/'))==digest
    checks=0
    def pack(row):
        value=row.float();s=value.abs().amax().clamp_min(1.e-8)/127.;q=torch.round(value/s).clamp(-127,127).to(torch.int8);return q,s
    for case in data['cases']:
        assert case['guards_passed']
        if case['prefix_elements'] in (0,256):assert case['integer_variants_equal'] and case['BF16_variants_equal'] and all(v['full_integer_FP32_reference_exact'] and v['full_output_tolerance_passed'] for v in case['variants'].values())
        else:
            assert case['prefix_elements']==16 and not case['variants']['static']['full_integer_FP32_reference_exact'] and case['variants']['dynamic']['full_integer_FP32_reference_exact']
            examples=torch.load(root/'prefix16_selected.pt',map_location='cpu',weights_only=True);assert len(examples)==32
            for row in examples:
                aq,asc=pack(row['original_a']);wq,wsc=pack(row['original_w'])
                assert torch.equal(aq,row['a_INT8']) and torch.equal(asc,row['ascale']) and torch.equal(wq,row['w_INT8']) and torch.equal(wsc,row['wscale'])
                exact=int(aq.to(torch.int64)@wq.to(torch.int64));assert exact==row['CPU_INT64']==row['dynamic_integer'] and exact!=row['static_integer'];checks+=1
            assert case['BF16_variant_mismatches']>0
    archive=dict(complete=True,diagnostic_only=True,independently_CPU_selected_rechecked=True,selected_bad_pairs_CPU_INT64=checks,selected_original_A_W_quantization_recomputed=True,
        original_sweep_complete=False,original_sweep_preserved=True,all_26_shapes_not_proven=True,
        zero_and_256_element_prefixes_aligned_cases_passed=True,offset16_static_baseline_incorrect=True,dynamic_reference_correct_at_all_three_tested_prefixes=True,
        exact_required_alignment_not_isolated=True,individual_pointer_constraints_not_isolated=True,new_kernel_not_adopted=True,raw_root=str(root),cases=data['cases'],frozen_sources=data['frozen_sources'],verifier_sha256=sha(__file__))
    (BASE/'M64_DIAGNOSTIC_GATE_32g.json').write_text(json.dumps(archive,indent=2)+'\n');print(json.dumps({k:v for k,v in archive.items() if k not in ('cases','frozen_sources')}))
if __name__=='__main__':main()
