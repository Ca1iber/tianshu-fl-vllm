# SPDX-License-Identifier: Apache-2.0
"""V-only software E5M2 byte storage; no engine hooks or model conversion."""
import ast
import hashlib
import math
from pathlib import Path
import struct

BASE = Path(__file__).resolve().parent
READER = BASE.parent / 'tianshu_k8v16_address32_20261020/qualified_S16_reader.py'
WRITER = BASE.parent / 'tianshu_soft8_mixed_planes_20261019/writer.py'
MODES = ('BF16_PV', 'FP16_PV')


def bf16_scalar(bits):
    return struct.unpack('>f', struct.pack('>I', bits << 16))[0]


def encode_bf16(bits):
    """Actual proposed bit algorithm after finite FP16 conversion (RNE)."""
    value = bf16_scalar(bits)
    magnitude = 0.0 if math.isnan(value) else min(abs(value), 57344.0)
    half = struct.unpack('<H', struct.pack('<e', magnitude))[0]
    code = (half + 127 + ((half >> 8) & 1)) >> 8
    assert 0 <= code <= 123
    return code | ((bits >> 8) & 128)


def reference_encode(value):
    """Independent real-number nearest-even E5M2 grid, without FP16 casts."""
    magnitude = abs(value)
    if math.isnan(value) or magnitude == 0:
        code = 0
    elif magnitude >= 57344:
        code = 123
    elif magnitude < 2.0 ** -14:
        code = round(magnitude * 65536.0)
    else:
        _, exponent = math.frexp(magnitude)
        exponent -= 1
        mantissa = round(magnitude / 2.0 ** (exponent - 2))
        if mantissa == 8:
            exponent += 1
            mantissa = 4
        code = min(123, (exponent + 15) * 4 + mantissa - 4)
    return code | (128 if math.copysign(1.0, value) < 0 else 0)


def reference_decode(code):
    magnitude = code & 127
    exponent, mantissa = divmod(magnitude, 4)
    if exponent == 31:
        value = math.inf if mantissa == 0 else math.nan
    elif exponent == 0:
        value = mantissa * 2.0 ** -16
    else:
        value = (1.0 + mantissa / 4.0) * 2.0 ** (exponent - 15)
    return math.copysign(value, -1.0 if code & 128 else 1.0)


def check_parent(path, raw):
    rows = (BASE / 'PARENT_SOURCES_32g.sha256').read_text().splitlines()
    expected = {str((BASE.parents[1] / row.split(None, 1)[1]).resolve()): row.split()[0]
                for row in rows}
    assert hashlib.sha256(raw).hexdigest() == expected[str(path.resolve())]


def render_reader(raw, mode):
    assert mode in MODES
    check_parent(READER, raw)
    original = raw.decode()
    before = 'V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)'
    after = 'V = _e5m2_v_cast(V_load, Q, v_scale, KV_QUANT_MODE)'
    anchor = '@triton.jit\ndef _cast_kv_tile('
    assert original.count(before) == original.count(anchor) == 1
    result = 'half.to(tl.bfloat16)' if mode == 'BF16_PV' else 'half'
    helper = f'''@triton.jit
def _e5m2_v_cast(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    if KV_QUANT_MODE == 0 and data.dtype == tl.uint8:
        half = (data.to(tl.uint16) << 8).to(tl.float16, bitcast=True)
        return {result}
    else:
        return _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE)


'''
    test = '''\n\n@triton.jit
def e5m2_decode_all(X, Y):
    i = tl.arange(0, 256)
    dummy_q = tl.full([256], 0, tl.bfloat16)
    tl.store(Y + i, _e5m2_v_cast(tl.load(X + i), dummy_q, None, 0))
'''
    generated = original.replace(before, after).replace(anchor, helper + anchor) + test
    assert generated.removesuffix(test).replace(helper, '').replace(after, before) == original
    def fn(source, name):
        return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ('_cast_kv_tile', 'reduce_segments', 'unified_attention'):
        assert ast.dump(fn(original, name)) == ast.dump(fn(generated, name))
    assert ast.dump(fn(original, 'kernel_unified_attention')) == ast.dump(fn(generated.replace(after, before), 'kernel_unified_attention'))
    return generated, dict(parent_sha256=hashlib.sha256(raw).hexdigest(),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(), mode=mode,
        Q_K_score_softmax_AST_unchanged=True, reducer_AST_unchanged=True,
        attention_AST_only_V_helper_call_changed=True,
        P_cast_dtype='BF16 unchanged' if mode == 'BF16_PV' else 'FP16 (explicit precision change)',
        finite_V_codes_exact_in_both_BF16_and_FP16=True)


def render_writer(raw):
    check_parent(WRITER, raw)
    original = raw.decode()
    before = 'value_out = _soft8_encode_bf16(value)'
    after = 'value_out = _e5m2_v_encode(value)'
    anchor = '@triton.jit\ndef _write('
    assert original.count(before) == original.count(anchor) == 1
    helper = '''@triton.jit
def _e5m2_v_encode(value):
    original_bits = value.to(tl.uint16, bitcast=True).to(tl.uint32)
    x = tl.abs(value.to(tl.float32))
    x = tl.where(x == x, tl.minimum(x, 57344.0), 0.0)
    half = x.to(tl.float16, fp_downcast_rounding="rtne")
    bits = half.to(tl.uint16, bitcast=True).to(tl.uint32)
    magnitude = (bits + 127 + ((bits >> 8) & 1)) >> 8
    return (magnitude | ((original_bits >> 8) & 128)).to(tl.uint8)


'''
    test = '''\n\n@triton.jit
def e5m2_encode_all(X, Y, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(Y + i, _e5m2_v_encode(tl.load(X + i, i < N, other=0)), i < N)
'''
    generated = original.replace(before, after).replace(anchor, helper + anchor) + test
    assert generated.removesuffix(test).replace(helper, '').replace(after, before) == original
    return generated, dict(parent_sha256=hashlib.sha256(raw).hexdigest(),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        only_V_encoder_changed=True, K_encoder_slot_address_and_negative_slot_rules_unchanged=True,
        rounding='nearest ties to even', saturation=57344, NaN='signed zero',
        infinity='signed saturation', reserved_codes_never_emitted=True)
