# SPDX-License-Identifier: Apache-2.0
"""Two independently typed cache planes; fixed software8 encoding unchanged."""
import torch
import triton
import triton.language as tl

@triton.jit
def _soft8_encode_bf16(data):
    bits = data.to(tl.uint16, bitcast=True).to(tl.uint32)
    magnitude = bits & 32767
    rounded = (magnitude + 7 + ((magnitude >> 4) & 1)) >> 4
    packed = tl.where(rounded >= 936, tl.minimum(rounded - 928, 127), 0)
    return (packed | ((bits & 32768) >> 8)).to(tl.uint8)


@triton.jit
def _write(Key, Value, K, V, Slots,
           KS: tl.constexpr, VS: tl.constexpr,
           KB: tl.constexpr, VB: tl.constexpr,
           KT: tl.constexpr, VT: tl.constexpr,
           KH: tl.constexpr, VH: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(Slots + row).to(tl.int64)
    if slot < 0:
        return
    offsets = tl.arange(0,256)
    head, dim = offsets // 128, offsets % 128
    key = tl.load(Key + row * KS + offsets)
    value = tl.load(Value + row * VS + offsets)
    if K.dtype.element_ty == tl.uint8:
        key_out = _soft8_encode_bf16(key)
    else:
        key_out = key
    if V.dtype.element_ty == tl.uint8:
        value_out = _soft8_encode_bf16(value)
    else:
        value_out = value
    block, token = slot // 16, slot % 16
    tl.store(K + block * KB + token * KT + head * KH + dim, key_out)
    tl.store(V + block * VB + token * VT + head * VH + dim, value_out)

@triton.jit
def encode_all(X,Y,N:tl.constexpr):
    i=tl.program_id(0)*256+tl.arange(0,256)
    tl.store(Y+i,_soft8_encode_bf16(tl.load(X+i,i<N,other=0)),i<N)

def write(key,value,k,v,slots):
    assert key.dtype==value.dtype==torch.bfloat16
    assert key.shape==value.shape and key.shape[1:]==(2,128)
    assert key.stride(1)==value.stride(1)==128 and key.stride(2)==value.stride(2)==1
    assert k.dtype in (torch.bfloat16,torch.uint8) and v.dtype in (torch.bfloat16,torch.uint8)
    assert k.shape==v.shape and k.shape[1:]==(16,2,128)
    assert k.stride(-1)==v.stride(-1)==1 and slots.shape==(key.shape[0],)
    return _write[(slots.numel(),)](key,value,k,v,slots,key.stride(0),value.stride(0),
        k.stride(0),v.stride(0),k.stride(1),v.stride(1),k.stride(2),v.stride(2),num_warps=1,num_stages=1)
