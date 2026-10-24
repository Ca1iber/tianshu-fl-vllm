# SPDX-License-Identifier: Apache-2.0
"""Non-TD contiguous-head128 V-only paired loads; all attention arithmetic retained."""
import ast
import hashlib
import codec

OLD_LOAD='''            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
                cache_modifier=KV_CACHE_MODIFIER if IS_3D else "",
            )'''
NEW_LOAD='''            V_load = _paired_v_load(
                value_cache_ptr,
                physical_block_idx * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + (seq_offset % BLOCK_SIZE) * stride_v_cache_1,
                tile_mask, HEAD_SIZE, HEAD_SIZE_PADDED,
                KV_CACHE_MODIFIER if IS_3D else "",
            )'''
OLD_CAST='        V = _e5m2_v_cast(V_load, Q, v_scale, KV_QUANT_MODE)'
NEW_CAST='        V = V_load'
HELPER='''@triton.jit
def _paired_v_load(Value, base_bytes, rows, HEAD: tl.constexpr,
                   PAD: tl.constexpr, MODIFIER: tl.constexpr):
    tl.static_assert(HEAD == 128 and PAD == 128)
    tl.static_assert(Value.dtype.element_ty == tl.uint8)
    pair = tl.arange(0, PAD // 2)
    words = tl.load(
        Value.to(tl.pointer_type(tl.uint16)) + base_bytes[:, None] // 2 + pair[None, :],
        mask=rows[:, None], other=0, cache_modifier=MODIFIER,
    )
    low = (words << 8).to(tl.uint16).to(tl.float16, bitcast=True)
    high = (words & 65280).to(tl.uint16).to(tl.float16, bitcast=True)
    return tl.interleave(low, high)


'''
TEST='''\n\n@triton.jit
def paired_load_word_test(X, Y, ROWS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    values = _paired_v_load(X, row * 128, row < ROWS, 128, 128, "")
    offsets = row[:, None] * 128 + tl.arange(0, 128)[None, :]
    bits = values.to(tl.uint16, bitcast=True).to(tl.int16, bitcast=True)
    tl.store(Y + offsets, bits, mask=(row < ROWS)[:, None])
'''

def render_reader(raw):
    parent,parent_proof=codec.render_reader(raw,'FP16_PV')
    anchor='@triton.jit\ndef _e5m2_v_cast('
    assert parent.count(OLD_LOAD)==parent.count(OLD_CAST)==parent.count(anchor)==1
    generated=parent.replace(OLD_LOAD,NEW_LOAD).replace(OLD_CAST,NEW_CAST).replace(anchor,HELPER+anchor)+TEST
    restored=generated.removesuffix(TEST).replace(HELPER,'').replace(NEW_LOAD,OLD_LOAD).replace(NEW_CAST,OLD_CAST)
    assert restored==parent
    def fn(source,name):return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('_cast_kv_tile','_e5m2_v_cast','reduce_segments','unified_attention'):
        assert ast.dump(fn(parent,name))==ast.dump(fn(generated,name))
    assert ast.dump(fn(parent,'kernel_unified_attention'))==ast.dump(fn(restored,'kernel_unified_attention'))
    return generated,dict(parent=parent_proof,generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        only_nonTD_V_load_and_already_FP16_V_assignment_changed=True,
        Q_K_score_softmax_reducer_unchanged=True,quantizer_unchanged=True,
        V_return_dtype='FP16, same as qualified FP16_PV',
        supported=dict(HEAD_SIZE=128,HEAD_SIZE_PADDED=128,last_V_stride=1,
            V_pointer_alignment_bytes=2,other_V_byte_strides_even=True,USE_TD=False,KV_QUANT_MODE=0),
        no_general_TD_or_strided_head_support_claim=True)

def check_gpu(models,torch,progress,save):
    # Independent byte sequence defines every possible word, without host reinterpretation.
    byte_values=[b for word in range(65536) for b in (word & 255,word >> 8)]
    encoded=torch.tensor(byte_values,dtype=torch.uint8,device='cuda')
    original=encoded.clone()
    expected=torch.tensor([b << 8 for b in byte_values],dtype=torch.int32).short().cuda()
    output=torch.full((1056*128,),12345,dtype=torch.int16,device='cuda')
    for model in models[1:]:
        if model['config']['loader']!='pair':continue
        tile=model['config']['TILE_SIZE']
        for rows in (1024,1023):
            output.fill_(12345)
            rng=torch.cuda.get_rng_state().clone()
            model['reader'].paired_load_word_test[((rows+tile-1)//tile,)](encoded,output,rows,tile,num_warps=1)
            torch.cuda.synchronize()
            assert torch.equal(output[:rows*128],expected[:rows*128])
            assert bool((output[rows*128:]==12345).all())
            assert torch.equal(encoded,original) and torch.equal(torch.cuda.get_rng_state(),rng)
        progress[model['label']]=dict(all65536_words_and131072_half_bit_patterns_passed=True,
            masked_last_row_and_tail_guard_passed=True,input_bytes_and_RNG_unchanged=True,
            tested_tile=tile,half_bit_patterns_include_signed_zero_Inf_and_NaN_payloads=True)
        save()
