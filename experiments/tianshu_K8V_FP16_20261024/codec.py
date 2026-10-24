# SPDX-License-Identifier: Apache-2.0
"""Native FP16 V storage/compute: same 12KiB page, unchanged software8 K."""
import ast
import hashlib
import math
from pathlib import Path
import struct

BASE = Path(__file__).resolve().parent
READER = BASE.parent / 'tianshu_k8v16_address32_20261020/qualified_S16_reader.py'
WRITER = BASE.parent / 'tianshu_soft8_mixed_planes_20261019/writer.py'
PARENTS = {
    str(READER): '7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106',
    str(WRITER): '27a74d3bdfee978780044eb69025c10a21d92a1b0a8cf567ee9fea1e0b8d12b5',
}

def bf16_scalar(bits):
    return struct.unpack('>f', struct.pack('>I', bits << 16))[0]

def encode_bf16(bits):
    value = bf16_scalar(bits)
    try:
        return struct.unpack('<H', struct.pack('<e', value))[0]
    except OverflowError:
        return 0xfc00 if bits & 0x8000 else 0x7c00

def reference_encode(value):
    sign = 0x8000 if math.copysign(1., value) < 0 else 0
    value = abs(value)
    if math.isnan(value): return sign | 0x7e00
    if math.isinf(value): return sign | 0x7c00
    if value < 2.**-14:
        return sign | round(value * 2.**24)
    exponent = math.frexp(value)[1] - 1
    mantissa = round(value * 2.**(10-exponent))
    if mantissa == 2048:
        exponent += 1
        mantissa = 1024
    if exponent > 15: return sign | 0x7c00
    return sign | ((exponent+15)<<10) | (mantissa-1024)

def reference_decode(bits):
    exponent, mantissa = (bits >> 10) & 31, bits & 1023
    value = (math.inf if not mantissa else math.nan) if exponent == 31 else (
        mantissa * 2.**-24 if exponent == 0 else (1+mantissa/1024) * 2.**(exponent-15))
    return math.copysign(value, -1. if bits & 0x8000 else 1.)

def checked(path, raw):
    assert hashlib.sha256(raw).hexdigest() == PARENTS[str(path)]
    return raw.decode()

def function(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)

def render_reader(raw, mode='FP16_PV'):
    assert mode == 'FP16_PV'
    original = checked(READER, raw)
    before = 'V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)'
    after = 'V = _native_v_cast(V_load, Q, v_scale, KV_QUANT_MODE)'
    anchor = '@triton.jit\ndef _cast_kv_tile('
    assert original.count(before) == original.count(anchor) == 1
    helper = '''@triton.jit
def _native_v_cast(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    if KV_QUANT_MODE == 0 and data.dtype == tl.float16:
        return data
    else:
        return _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE)


'''
    test = '''

@triton.jit
def fp16_decode_all(X, Y, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    dummy_q = tl.full([256], 0, tl.bfloat16)
    half = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, _native_v_cast(half, dummy_q, None, 0), i < N)
'''
    generated = original.replace(before, after).replace(anchor, helper + anchor) + test
    assert generated.removesuffix(test).replace(helper, '').replace(after, before) == original
    for name in ('_cast_kv_tile', 'reduce_segments', 'unified_attention'):
        assert ast.dump(function(original, name)) == ast.dump(function(generated, name))
    assert ast.dump(function(original, 'kernel_unified_attention')) == ast.dump(function(generated.replace(after,before), 'kernel_unified_attention'))
    return generated, dict(parent_sha256=hashlib.sha256(raw).hexdigest(),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        Q_K_score_softmax_AST_unchanged=True, reducer_AST_unchanged=True,
        attention_AST_only_V_helper_call_changed=True,
        P_cast_dtype='FP16; explicit precision change', V_storage_dtype='native FP16',
        physical_page_bytes=12288, extra_byte_decode=False)

def render_writer(raw):
    original = checked(WRITER, raw)
    before = 'value_out = value'
    after = 'value_out = _native_v_store_cast(value, V.dtype.element_ty == tl.float16)'
    guard_before = 'v.dtype in (torch.bfloat16,torch.uint8)'
    guard_after = 'v.dtype in (torch.bfloat16,torch.uint8,torch.float16)'
    anchor = '@triton.jit\ndef _write('
    assert original.count(before) == original.count(anchor) == original.count(guard_before) == 1
    helper = '''@triton.jit
def _native_v_store_cast(value, IS_HALF: tl.constexpr):
    if IS_HALF:
        return value.to(tl.float32).to(tl.float16, fp_downcast_rounding="rtne")
    else:
        return value


'''
    test = '''

@triton.jit
def fp16_encode_all(X, Y, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(Y + i, _native_v_store_cast(tl.load(X + i, i < N, other=0), True), i < N)
'''
    generated = original.replace(before, after).replace(guard_before,guard_after).replace(anchor,helper+anchor)+test
    assert generated.removesuffix(test).replace(helper,'').replace(after,before).replace(guard_after,guard_before) == original
    return generated, dict(parent_sha256=hashlib.sha256(raw).hexdigest(),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        K_encoder_slot_address_and_negative_slot_rules_unchanged=True,
        V_cast_only_for_FP16_destination=True, rounding='nearest ties to even',
        overflow='signed infinity; unsaturated IEEE FP16', NaN='NaN classification; payload unspecified')
