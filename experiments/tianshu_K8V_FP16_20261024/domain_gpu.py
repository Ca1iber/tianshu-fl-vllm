# SPDX-License-Identifier: Apache-2.0
"""Full native-half writer/decode domains. Imported only after GPU idle check."""
def validate(codec, models, candidate_writer, control_writer, torch):
    bits=torch.arange(65536,dtype=torch.int32).short()
    x=bits.view(torch.bfloat16).cuda()
    expected=torch.tensor([codec.encode_bf16(i) for i in range(65536)],dtype=torch.int32,device='cuda')
    actual=torch.empty_like(x,dtype=torch.float16)
    candidate_writer.fp16_encode_all[(256,)](x,actual,65536,num_warps=1)
    actual_bits=actual.view(torch.int16).int()&65535
    nonnan=~torch.isnan(x)
    assert torch.equal(actual_bits[nonnan],expected[nonnan])
    assert torch.equal(torch.isnan(actual),torch.isnan(x))
    bulk=x.half()
    assert torch.equal(actual_bits[nonnan],(bulk.view(torch.int16).int()&65535)[nonnan])
    assert torch.equal(torch.isnan(actual),torch.isnan(bulk))
    k_control=torch.empty(65536,dtype=torch.uint8,device='cuda')
    k_candidate=torch.empty_like(k_control)
    control_writer.encode_all[(256,)](x,k_control,65536,num_warps=1)
    candidate_writer.encode_all[(256,)](x,k_candidate,65536,num_warps=1)
    assert torch.equal(k_control,k_candidate)
    codes=bits.view(torch.float16).cuda()
    decoded_ref=torch.tensor([codec.reference_decode(i) for i in range(65536)],dtype=torch.float32,device='cuda')
    finite=torch.isfinite(decoded_ref)
    for model in models[1:]:
        decoded=torch.empty(65536,dtype=torch.float32,device='cuda')
        model['reader'].fp16_decode_all[(256,)](codes,decoded,65536,num_warps=1)
        assert torch.equal(decoded[finite],decoded_ref[finite])
        assert torch.equal(torch.signbit(decoded[finite]),torch.signbit(decoded_ref[finite]))
        assert torch.equal(torch.isnan(decoded),torch.isnan(decoded_ref))
        assert torch.equal(torch.isinf(decoded),torch.isinf(decoded_ref))
        inf=torch.isinf(decoded_ref)
        assert torch.equal(torch.signbit(decoded[inf]),torch.signbit(decoded_ref[inf]))
    return dict(complete=True,all65536_BF16_writer_encodings_match_independent_CPU_except_NaN_payload=True,
        all65536_K_encodings_bitwise_unchanged=True,GPU_bulk_fixture_encoder_matches_all_patterns_or_NaN_class=True,
        all65536_half_decodes_values_signs_or_NaN_class_match_all_readers=True,
        half_subnormal_values_not_flushed=True,signedzero_and_overflow_inf_checked=True)
